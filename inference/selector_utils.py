"""
selector_utils.py

Shared utilities for frame selector evaluation scripts.
Provides:
  - build_projector: construct correct projector (identity/linear/mlp) from checkpoint config
  - aggregate_*: aggregation functions matching training code
  - load_trained_model_v2: loads checkpoint with correct projector type and aggregation config
  - apply_aggregation: convenience wrapper to apply the right aggregation to embeddings
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

from utils import InternVLWithHead, MLPProjector
from attention_patch import apply_debug_patch as enable_qwen2_pos_shift_attention


# ──────────────────────────────────────────────────────────
# 1. Projector builders
# ──────────────────────────────────────────────────────────

def build_projector(ptype, hidden_size, projection_dim):
    """Build projector matching training-time configuration."""
    if ptype == 'identity':
        return nn.Identity()
    elif ptype == 'linear':
        return nn.Linear(hidden_size, projection_dim, bias=False)
    else:
        return MLPProjector(hidden_size, projection_dim)


# ──────────────────────────────────────────────────────────
# 2. Aggregation functions (mirrors train_framescout_ablation.py)
# ──────────────────────────────────────────────────────────

def aggregate_none(h):
    return h


def aggregate_successor(h, beta):
    """z[i] = h[i] + beta*z[i+1], exponential future discount."""
    n = h.shape[0]
    out = [None] * n
    out[n - 1] = h[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = h[i] + beta * out[i + 1]
    return torch.stack(out, dim=0)


def aggregate_local_mean(h, window):
    """z[i] = mean(h[i : i+window]) (causal future)."""
    n, d = h.shape
    out = torch.zeros_like(h)
    for i in range(n):
        end = min(i + window, n)
        out[i] = h[i:end].mean(dim=0)
    return out


def aggregate_future_mean(h):
    """z[i] = cummean(h[i:])."""
    n, d = h.shape
    h_flip = torch.flip(h, dims=[0])
    cum = torch.cumsum(h_flip, dim=0)
    counts = torch.arange(1, n + 1, device=h.device, dtype=h.dtype).unsqueeze(1)
    return torch.flip(cum / counts, dims=[0])


def aggregate_future_max(h):
    """z[i] = cummax(h[i:])."""
    n, d = h.shape
    h_flip = torch.flip(h, dims=[0])
    out_flip, _ = torch.cummax(h_flip, dim=0)
    return torch.flip(out_flip, dims=[0])


AGGREGATORS = {
    'none': aggregate_none,
    'successor': aggregate_successor,
    'local_mean': aggregate_local_mean,
    'future_mean': aggregate_future_mean,
    'future_max': aggregate_future_max,
}


def apply_aggregation(embeddings, agg_config):
    """
    Apply the configured aggregation to chunk embeddings.

    Args:
        embeddings: [N, D] tensor of raw projected chunk embeddings
        agg_config: dict with keys 'aggregation', 'beta', 'local_window'
    Returns:
        [N, D] tensor of aggregated embeddings
    """
    agg_type = agg_config.get('aggregation', 'none')
    if agg_type == 'none':
        return embeddings
    elif agg_type == 'successor':
        return aggregate_successor(embeddings, agg_config.get('beta', 0.3))
    elif agg_type == 'local_mean':
        return aggregate_local_mean(embeddings, agg_config.get('local_window', 3))
    elif agg_type == 'future_mean':
        return aggregate_future_mean(embeddings)
    elif agg_type == 'future_max':
        return aggregate_future_max(embeddings)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


# ──────────────────────────────────────────────────────────
# 3. Config loading
# ──────────────────────────────────────────────────────────

def infer_config_from_name(checkpoint_dir):
    """Fallback: infer aggregation config from directory name."""
    name = os.path.basename(checkpoint_dir.rstrip('/'))
    config = {
        'projector': 'mlp',
        'aggregation': 'none',
        'beta': 0.0,
        'local_window': 3,
    }
    # Parse aggregation
    if 'successor' in name.lower() or 'revaccum' in name.lower() or 'revAccum' in name.lower():
        config['aggregation'] = 'successor'
    elif 'none' in name or '_mlp_ctr_none' in name or 'e3_none' in name:
        config['aggregation'] = 'none'
    # Parse beta
    import re
    beta_match = re.search(r'beta[=_]?([0-9.]+)', name, re.IGNORECASE)
    if beta_match:
        config['beta'] = float(beta_match.group(1))
    elif config['aggregation'] == 'successor':
        config['beta'] = 0.3  # default
    # Parse projector
    if 'linear' in name.lower():
        config['projector'] = 'linear'
    elif 'identity' in name.lower():
        config['projector'] = 'identity'
    return config


def load_config(checkpoint_dir):
    """Load run_config.json if present, otherwise infer from directory name.
    Walks up to 4 levels to find run_config.json or an informative directory name."""
    # Try exact path
    config_path = os.path.join(checkpoint_dir, 'run_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    # Walk up to find run_config.json
    current = checkpoint_dir
    for _ in range(4):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
        config_path = os.path.join(current, 'run_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    # Fallback: infer from directory chain
    # Try each level of the path for config hints
    parts = checkpoint_dir.rstrip('/').split('/')
    for i in range(len(parts) - 1, -1, -1):
        config = infer_config_from_name(parts[i])
        if config.get('aggregation') != 'none' or config.get('projector') != 'mlp':
            return config
    return infer_config_from_name(checkpoint_dir)


# ──────────────────────────────────────────────────────────
# 3b. Simple model loading for revaccum selectors (no agg_config needed)
# ──────────────────────────────────────────────────────────

def load_model_with_correct_projector(device, model_path):
    """
    Lightweight version for revaccum selectors.
    Loads model with correct projector type (mlp/linear/identity).

    Uses manual model construction to avoid transformers 5.x compatibility issues
    with custom model code.
    """
    import os as _os

    # Load config from checkpoint dir
    config = load_config(model_path)
    projector_type = config.get('projector', 'mlp')
    projection_dim = config.get('projection_dim', 1024)

    print(f"[RevAccum] Config: projector={projector_type}, dim={projection_dim}")

    # Use local patched model path for tokenizer and config
    base_model_path = config.get('model_path', None)
    if base_model_path is None or not _os.path.exists(_os.path.join(base_model_path, 'config.json')):
        base_model_path = os.environ.get("FRAMESCOUT_BASE_MODEL", "OpenGVLab/InternVL2_5-1B")

    print(f"[RevAccum] Base model from: {base_model_path}")

    # ---- Manual model construction (avoids from_pretrained issues with transformers 5.x) ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    # Import the model class from local_internvl (has compatibility fixes)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from local_internvl.modeling_internvl_chat import InternVLChatModel

    base_model = InternVLChatModel(model_config)
    hidden_size = base_model.language_model.config.hidden_size

    model = InternVLWithHead(base_model, hidden_size, projection_dim).to(device, dtype=torch.bfloat16)
    model.img_projector = build_projector(projector_type, hidden_size, projection_dim).to(
        device=device, dtype=torch.bfloat16)
    model.txt_projector = build_projector(projector_type, hidden_size, projection_dim).to(
        device=device, dtype=torch.bfloat16)

    enable_qwen2_pos_shift_attention(model.model.language_model)

    # Find and load checkpoint
    checkpoint_path = _os.path.join(model_path, "pytorch_model.bin")
    if not _os.path.exists(checkpoint_path):
        for root, dirs, files in _os.walk(model_path):
            for f in files:
                if f == 'pytorch_model.bin':
                    checkpoint_path = _os.path.join(root, f)
                    break
            if _os.path.exists(checkpoint_path):
                break

    if _os.path.exists(checkpoint_path):
        print(f"[RevAccum] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"[RevAccum] Warning: No pytorch_model.bin found, using base model.")

    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────
# 4. Model loading (replaces load_trained_model in selectors)
# ──────────────────────────────────────────────────────────

def load_trained_model_v2(device, model_path):
    """
    Load a trained FrameScout checkpoint with correct projector type and
    aggregation settings.

    Reads run_config.json (or infers from directory name) to determine:
      - projector type (identity / linear / mlp)
      - aggregation strategy
      - beta / local_window parameters

    Args:
        device: torch device
        model_path: path to the model directory (can be a full checkpoint path
                    like .../epoch_2/ or the top-level experiment directory)

    Returns:
        (model, tokenizer, agg_config)
        agg_config: dict with keys 'aggregation', 'beta', 'local_window'
    """
    print(f"Loading model from: {model_path}")

    # Determine checkpoint directory and config
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        # model_path might be the top-level experiment dir; find the actual checkpoint
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if f == 'pytorch_model.bin':
                    checkpoint_path = os.path.join(root, f)
                    break
            if os.path.exists(checkpoint_path):
                break

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in or under {model_path}")

    actual_checkpoint_dir = os.path.dirname(checkpoint_path)
    print(f"Checkpoint file: {checkpoint_path}")

    # Load config
    config = load_config(actual_checkpoint_dir)
    # Also check parent dir
    if 'projector' not in config or 'aggregation' not in config:
        parent_config = load_config(os.path.dirname(actual_checkpoint_dir))
        config = {**parent_config, **config}  # child overrides parent

    print(f"Config: projector={config.get('projector','?')}, "
          f"aggregation={config.get('aggregation','?')}, "
          f"beta={config.get('beta','?')}")

    # Load base model from local patched model path (avoids transformers 5.x compatibility)
    base_model_path = config.get('model_path', None)
    if base_model_path is None or not os.path.exists(os.path.join(base_model_path, 'config.json')):
        base_model_path = os.environ.get("FRAMESCOUT_BASE_MODEL", "OpenGVLab/InternVL2_5-1B")

    print(f"Base model from: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    base_model = AutoModel.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_flash_attn=False
    )

    hidden_size = base_model.language_model.config.hidden_size
    projection_dim = config.get('projection_dim', 1024)
    projector_type = config.get('projector', 'mlp')

    model = InternVLWithHead(base_model, hidden_size, projection_dim).to(device, dtype=torch.bfloat16)

    # Replace projectors with correct type
    model.img_projector = build_projector(projector_type, hidden_size, projection_dim).to(
        device=device, dtype=torch.bfloat16)
    model.txt_projector = build_projector(projector_type, hidden_size, projection_dim).to(
        device=device, dtype=torch.bfloat16)

    enable_qwen2_pos_shift_attention(model.model.language_model)

    # Load checkpoint weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    agg_config = {
        'aggregation': config.get('aggregation', 'none'),
        'beta': config.get('beta', 0.0),
        'local_window': config.get('local_window', 3),
    }

    return model, tokenizer, agg_config
