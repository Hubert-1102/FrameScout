import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from tqdm import tqdm
import json
import hashlib
from datasets import load_dataset
import time
import shutil
import argparse
import re

# Reuse MLPProjector / attention_patch from the original streaming module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MLPProjector
from selector_utils import load_model_with_correct_projector
from attention_patch import apply_debug_patch as enable_qwen2_pos_shift_attention


def natural_key(string_):
    """Natural sort (1,2,10 instead of 1,10,2)."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]


# ================= Static config =================
FRAMES_ROOT_PATH = os.environ.get("VIDEOMME_FRAMES", "/path/to/videomme/frames1024")
DATASET_PATH = os.environ.get("VIDEOMME_DATA", "/path/to/videomme")

GPUS_TO_USE = 8
STAGE1_PROCS_PER_GPU = 4   # balanced between speed and I/O contention
RUN_MODE = 'all'  # 'compute_emb', 'select_frames', 'all'
# =================================================


# ==========================================
# Single-direction wrapper (matches train_script_two_stage.py and
# train_script_two_stage_reverse_accum.py):
#   - img_projector: hidden_size     -> projection_dim   (NOT 2*hidden_size)
#   - txt_projector: hidden_size     -> projection_dim
# ==========================================
class InternVLWithHead(nn.Module):
    def __init__(self, original_model, hidden_size, output_size):
        super().__init__()
        self.model = original_model
        self.img_projector = MLPProjector(hidden_size, output_size)
        self.txt_projector = MLPProjector(hidden_size, output_size)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ==========================================
# Streaming KV-cache manager
# ==========================================
class MemoryManager:
    def __init__(self):
        self.system_len = 0
        self.last_round_len = 0

    def set_system_len(self, length):
        self.system_len = length

    def get_seq_length(self, past_key_values):
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return past_key_values.get_seq_length()
            except:
                return past_key_values.get_seq_length(0)
        elif isinstance(past_key_values, (tuple, list)):
            return past_key_values[0][0].shape[2]
        return 0

    def prune_cache(self, past_key_values):
        if past_key_values is None:
            return None
        current_len = self.get_seq_length(past_key_values)
        keep_len = self.system_len + self.last_round_len
        if current_len <= keep_len:
            return past_key_values

        new_cache = DynamicCache()
        num_layers = len(past_key_values) if isinstance(past_key_values, (tuple, list)) else len(past_key_values.layers)

        for i in range(num_layers):
            try:
                layer_content = past_key_values[i]
                if isinstance(layer_content, (tuple, list)):
                    k, v = layer_content
                else:
                    k, v = layer_content.prev_k, layer_content.prev_v
            except:
                break
            k_new = torch.cat([k[:, :, :self.system_len, :], k[:, :, -self.last_round_len:, :]], dim=2)
            v_new = torch.cat([v[:, :, :self.system_len, :], v[:, :, -self.last_round_len:, :]], dim=2)
            new_cache.update(k_new, v_new, i)
        new_cache._seen_tokens = keep_len
        return new_cache


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def process_one_frame(image, input_size=448):
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)


def load_trained_model(device, model_path):
    print(f"[RevAccum] Loading model from: {model_path}")
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    base_model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_flash_attn=False
    )
    hidden_size = base_model.language_model.config.hidden_size
    # Single-direction projector -> reverse-accumulation runs BEFORE projection
    model = InternVLWithHead(base_model, hidden_size, 1024).to(device, dtype=torch.bfloat16)
    enable_qwen2_pos_shift_attention(model.model.language_model)

    if os.path.exists(checkpoint_path):
        print(f"[RevAccum] Loading custom checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"[RevAccum] Warning: Custom checkpoint not found at {checkpoint_path}, using base model weights.")

    model.eval()
    return model, tokenizer


def get_text_embeddings(model, tokenizer, texts, device):
    prompts = [f"<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n" for text in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.language_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1

        target_embeddings = last_hidden_state[torch.arange(len(texts), device=device), last_token_indices]
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        txt_emb = model.txt_projector(target_embeddings)
        return txt_emb.detach()


# ==========================================
# Streaming per-frame hidden extraction (forward order only).
# Returns (N, hidden_size) aligned to the INPUT image order.
# ==========================================
def _streaming_frame_hidden_states(model, tokenizer, images, device, chunk_size=8):
    hiddens = []
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    if not hasattr(model.model, "img_context_token_id") or model.model.img_context_token_id is None:
        model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    img_context_token_id = model.model.img_context_token_id
    img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    newline_ids = tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    mem_manager = MemoryManager()
    prefix_prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n"
    prefix_tokens = tokenizer(prefix_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.language_model(input_ids=prefix_tokens, use_cache=True)
        past_key_values = outputs.past_key_values

    mem_manager.set_system_len(prefix_tokens.shape[1])
    global_seq_len = prefix_tokens.shape[1]

    for i in range(0, len(images), chunk_size):
        batch_imgs = images[i: i + chunk_size]
        past_key_values = mem_manager.prune_cache(past_key_values)

        batch_pv = []
        batch_ids = []

        for img in batch_imgs:
            pv = process_one_frame(img).to(device, dtype=torch.bfloat16)
            batch_pv.append(pv)
            num_patches = pv.shape[0]
            img_tokens = [img_start_id] + [img_context_token_id] * (model.model.num_image_token * num_patches) + [img_end_id]
            batch_ids.append(torch.cat([torch.tensor([img_tokens], device=device), newline_ids], dim=1))

        pixel_values = torch.cat(batch_pv, dim=0)
        input_ids = torch.cat(batch_ids, dim=1)

        current_phys_len = mem_manager.get_seq_length(past_key_values)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(global_seq_len, global_seq_len + seq_len, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones((1, current_phys_len + seq_len), dtype=torch.long, device=device)
        image_flags = torch.ones((pixel_values.shape[0], 1), device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids, pixel_values=pixel_values, past_key_values=past_key_values,
                image_flags=image_flags, position_ids=position_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=True
            )
            past_key_values = outputs.past_key_values
            global_seq_len += seq_len
            mem_manager.last_round_len = seq_len

            last_hidden = outputs.hidden_states[-1]
            end_indices = torch.nonzero(input_ids[0] == img_end_id, as_tuple=True)[0]
            chunk_hiddens = last_hidden[0, end_indices, :]  # (chunk_len, hidden)
            hiddens.append(chunk_hiddens)

    if not hiddens:
        return torch.tensor([]).to(device)
    return torch.cat(hiddens, dim=0)



# ==========================================
# Aggregation variants (for E5 ablation eval)
#   Matches training-side aggregate_local_mean / aggregate_future_mean
#   in train_framescout_ablation.py (applied BEFORE projector).
# ==========================================
def aggregate_local_mean(h, window):
    """z[i] = mean(h[i : i+window]) (causal future)."""
    n = h.shape[0]
    out = torch.empty_like(h)
    for i in range(n):
        end = min(i + window, n)
        out[i] = h[i:end].mean(dim=0)
    return out

def aggregate_future_mean(h):
    """z[i] = cummean(h[i:])."""
    n = h.shape[0]
    h_flip = torch.flip(h, dims=[0])
    cum = torch.cumsum(h_flip, dim=0)
    counts = torch.arange(1, n + 1, device=h.device, dtype=h.dtype).unsqueeze(1)
    return torch.flip(cum / counts, dims=[0])

def apply_aggregation(h, beta):
    """Select aggregation based on env vars FRAMESCOUT_AGGREGATION / FRAMESCOUT_LOCAL_WINDOW.
    Defaults to successor (reverse_accumulate) for backward compatibility."""
    import os
    agg = os.environ.get('FRAMESCOUT_AGGREGATION', 'successor')
    if agg == 'successor':
        return reverse_accumulate(h, beta)
    elif agg == 'local_mean':
        w = int(os.environ.get('FRAMESCOUT_LOCAL_WINDOW', '3'))
        return aggregate_local_mean(h, w)
    elif agg == 'future_mean':
        return aggregate_future_mean(h)
    elif agg == 'none':
        return h
    else:
        raise ValueError(f'Unknown aggregation: {agg}')

# ==========================================
# Reverse Accumulation (same as training):
#   h'[N-1] = h[N-1]
#   h'[i]   = h[i] + beta * h'[i+1]
# Expands to: h'[i] = sum_{k>=i} beta^(k-i) * h[k]
# ==========================================
def reverse_accumulate(h, beta):
    if h is None or h.shape[0] == 0:
        return h
    n = h.shape[0]
    # Build from the back, in-place along a fresh tensor to keep memory tidy
    out = torch.empty_like(h)
    out[n - 1] = h[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = h[i] + beta * out[i + 1]
    return out


# ==========================================
# RevAccum streaming video embedding:
#   1) forward streaming -> h_fwd (N, H)
#   2) reverse_accumulate(h_fwd, beta) -> h_ra (N, H)
#   3) L2-normalize -> img_projector -> (N, D)
# ==========================================
def get_revaccum_streaming_video_embeddings(model, tokenizer, images, device, chunk_size=8, beta=0.5):
    if len(images) == 0:
        return torch.tensor([]).to(device)

    h_fwd = _streaming_frame_hidden_states(model, tokenizer, images, device, chunk_size=chunk_size)
    if h_fwd.numel() == 0:
        return torch.tensor([]).to(device)

    h_ra = apply_aggregation(h_fwd, beta)            # (N, H) [aggregation from env]
    h_ra = F.normalize(h_ra, dim=-1)                # match training
    return model.img_projector(h_ra)                # (N, projection_dim)


# ==========================================
# I/O utilities
# ==========================================
def load_frames_from_dir(frames_dir):
    if not os.path.exists(frames_dir):
        return None, None
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = sorted(
        [f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in valid_exts],
        key=natural_key
    )
    if not files:
        return None, None

    pil_images = []
    file_names = []
    for f in files:
        try:
            img_path = os.path.join(frames_dir, f)
            img = Image.open(img_path).convert('RGB')
            pil_images.append(img)
            file_names.append(f)
        except Exception:
            continue
    return file_names, pil_images


def generate_unique_folder_name(video_id, question_text):
    q_hash = hashlib.md5(question_text.encode('utf-8')).hexdigest()[:8]
    safe_video_id = video_id.replace('/', '_').replace('\\', '_')
    return f"{safe_video_id}_{q_hash}"


# ==========================================
# Stage 1: Embedding pre-computation worker
# ==========================================
def worker_precompute_embeddings(rank, local_world_size, queue, node_video_ids,
                                 model_path, chunk_size, video_emb_dir, beta):
    gpu_id = rank % GPUS_TO_USE
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    print(f"[Stage 1 - Worker {rank} / GPU {gpu_id}] Init RevAccum Model for Embedding Pre-computation (beta={beta})...")

    try:
        model, tokenizer = load_model_with_correct_projector(device, model_path)
    except Exception as e:
        print(f"[Worker {rank} / GPU {gpu_id}] Model loading failed: {e}")
        return

    my_video_ids = node_video_ids[rank::local_world_size]

    for video_id in my_video_ids:
        try:
            safe_vid = video_id.replace('/', '_')
            save_path = os.path.join(video_emb_dir, f"{safe_vid}.pt")

            if os.path.exists(save_path):
                try:
                    _ = torch.load(save_path, map_location='cpu')
                    queue.put(1)
                    continue
                except:
                    print(f"[Worker {rank} / GPU {gpu_id}] Corrupted embedding for {video_id}, recomputing...")

            current_video_frames_dir = os.path.join(FRAMES_ROOT_PATH, video_id)
            if not os.path.exists(current_video_frames_dir):
                queue.put(1)
                continue

            file_names, images = load_frames_from_dir(current_video_frames_dir)
            if not images:
                queue.put(1)
                continue

            img_embs = get_revaccum_streaming_video_embeddings(
                model, tokenizer, images, device, chunk_size=chunk_size, beta=beta
            )
            torch.save(img_embs.cpu(), save_path)

            queue.put(1)

        except Exception as e:
            print(f"[Worker {rank} / GPU {gpu_id}] Error processing video {video_id}: {e}")
            queue.put(1)


# ==========================================
# Stage 2: Selection & saving
# ==========================================
def select_smart_indices_with_nms(scores, existing_indices, num_smart, min_distance=5):
    total_frames = len(scores)
    work_scores = scores.copy()
    work_scores[existing_indices] = -np.inf

    selected_smart_indices = []

    for _ in range(num_smart):
        best_idx = np.argmax(work_scores)
        if work_scores[best_idx] == -np.inf:
            break
        selected_smart_indices.append(best_idx)

        start = max(0, best_idx - min_distance)
        end = min(total_frames, best_idx + min_distance + 1)
        work_scores[start:end] = -np.inf

    if len(selected_smart_indices) < num_smart:
        needed = num_smart - len(selected_smart_indices)
        backup_scores = scores.copy()
        backup_scores[existing_indices] = -np.inf
        if len(selected_smart_indices) > 0:
            backup_scores[np.array(selected_smart_indices)] = -np.inf

        valid_indices = np.where(backup_scores > -np.inf)[0]
        if len(valid_indices) > 0:
            if len(valid_indices) <= needed:
                remaining_best = valid_indices
            else:
                remaining_best = np.argpartition(backup_scores, -needed)[-needed:]
            selected_smart_indices.extend(remaining_best)

    return np.array(selected_smart_indices)


def worker_select_frames(rank, local_world_size, queue, node_dataset_indices,
                         model_path, output_frames_path, num_uniform, num_smart,
                         video_emb_dir, nms_ratio):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Stage 2 - Rank {rank}] Init RevAccum Model for Text Query...")

    try:
        model, tokenizer = load_model_with_correct_projector(device, model_path)
    except Exception as e:
        print(f"[Rank {rank}] Model loading failed: {e}")
        return

    if DATASET_PATH.endswith('.json'):
        full_dataset = load_dataset("json", data_files=DATASET_PATH, split="test")
    else:
        full_dataset = load_dataset(DATASET_PATH, split="test")

    my_indices = node_dataset_indices[rank::local_world_size]

    total_frames_needed = num_uniform + num_smart

    for idx in my_indices:
        try:
            item = full_dataset[idx]
            video_id = item.get('videoID')
            question = item['question']
            options = item['options']
            ground_truth = item.get('answer', None)

            if not video_id:
                queue.put(1)
                print('error: missing videoID')
                continue

            unique_folder_name = generate_unique_folder_name(video_id, question)
            save_dir = os.path.join(output_frames_path, unique_folder_name)

            safe_vid = video_id.replace('/', '_')
            emb_path = os.path.join(video_emb_dir, f"{safe_vid}.pt")

            if not os.path.exists(emb_path):
                queue.put(1)
                continue

            img_embs = torch.load(emb_path, map_location=device).to(dtype=torch.bfloat16)
            total_frames_emb = img_embs.shape[0]

            candidate_indices_local = np.arange(total_frames_emb)
            final_local_indices = []

            if total_frames_emb <= total_frames_needed:
                final_local_indices = candidate_indices_local
            else:
                combined_query_text = f"Question: {question}\nOptions: {options}\n"
                with torch.no_grad():
                    txt_emb = get_text_embeddings(model, tokenizer, [combined_query_text], device)
                    scores = torch.matmul(F.normalize(img_embs, dim=-1), txt_emb.t()).squeeze()
                scores = scores.detach().float().cpu().numpy()
                if scores.ndim == 0:
                    scores = np.array([scores])

                uniform_local_indices = np.linspace(0, total_frames_emb - 1, num_uniform).astype(int) \
                    if num_uniform > 0 else np.array([], dtype=int)

                nms_radius = int(total_frames_emb * nms_ratio)
                nms_radius = max(nms_radius, 8)
                print(f'num_radius:  {nms_radius}')

                smart_local_indices = select_smart_indices_with_nms(
                    scores,
                    uniform_local_indices,
                    num_smart,
                    min_distance=nms_radius
                )
                final_local_indices = np.concatenate([uniform_local_indices, smart_local_indices])
                final_local_indices = np.sort(final_local_indices)

            current_video_frames_dir = os.path.join(FRAMES_ROOT_PATH, video_id)
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            all_files = sorted(
                [f for f in os.listdir(current_video_frames_dir) if os.path.splitext(f)[1].lower() in valid_exts],
                key=natural_key
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            frame_mapping = []
            for i, idx2 in enumerate(final_local_indices):
                if idx2 < len(all_files):
                    original_name = all_files[idx2]
                    new_name = f"{i+1:03d}.jpg"
                    frame_mapping.append({
                        "new_name": new_name,
                        "original_name": original_name,
                        "original_index": int(idx2)
                    })

            info_path = os.path.join(save_dir, "query_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_id": video_id,
                    "question": question,
                    "options": options,
                    "answer": ground_truth,
                    "selected_frames": frame_mapping
                }, f, indent=2, ensure_ascii=False)

            for m in frame_mapping:
                src_path = os.path.join(current_video_frames_dir, m['original_name'])
                dst_path = os.path.join(save_dir, m['new_name'])
                try:
                    shutil.copy(src_path, dst_path)
                except Exception:
                    pass

            queue.put(1)

        except Exception as e:
            print(f"[Rank {rank}] Error processing query idx {idx}: {e}")
            queue.put(1)

    print(f"[Rank {rank}] Finished.")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-node Distributed Video Processing (Reverse-Accumulation Streaming)")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the (reverse-accumulation) model directory")
    parser.add_argument("--output_frames_path", type=str, required=True, help="Root path for saving selected frames")
    parser.add_argument("--video_embeddings_path", type=str, required=True, help="Root path for saving video embeddings")

    parser.add_argument("--num_uniform", type=int, default=0, help="Number of uniform frames")
    parser.add_argument("--num_smart", type=int, default=60, help="Number of smart selected frames")
    parser.add_argument("--nms_ratio", type=float, default=0.04,
                        help="NMS suppression radius as a ratio of candidate frames")
    parser.add_argument("--chunk_size", type=int, default=12, help="Chunk size for streaming embedding")
    parser.add_argument("--stage1_procs_per_gpu", type=int, default=STAGE1_PROCS_PER_GPU,
                        help="Number of Stage 1 embedding worker processes per GPU")

    # Reverse-accumulation decay factor (must match training)
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Reverse accumulation decay factor, h'[i] = h[i] + beta * h'[i+1]")

    parser.add_argument("--world_size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Current node index")

    parser.add_argument("--subset_file", type=str, default=None,
                        help="Optional path to a JSON whitelist ({'video_ids': [...]}); "
                             "if provided, only videos in the whitelist are processed.")

    args = parser.parse_args()
    if args.nms_ratio < 0:
        raise ValueError("--nms_ratio must be non-negative")

    world_size = int(os.getenv("WORLD_SIZE", args.world_size))
    rank = int(os.getenv("RANK", args.node_rank))

    print("==================================================")
    print(f"[RevAccum Streaming] Distributed Setup: Node {rank + 1} / {world_size}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Path: {args.output_frames_path}")
    print(f"Beta: {args.beta}")
    print(f"Params: Uniform={args.num_uniform}, Smart={args.num_smart}, "
          f"Chunk={args.chunk_size}, NMSRatio={args.nms_ratio}")
    print(f"Stage 1 Workers: {GPUS_TO_USE} GPUs x {args.stage1_procs_per_gpu} procs/GPU = {GPUS_TO_USE * args.stage1_procs_per_gpu}")
    print("==================================================")

    mp.set_start_method('spawn', force=True)
    VIDEO_EMB_DIR = args.video_embeddings_path
    if not os.path.exists(VIDEO_EMB_DIR):
        os.makedirs(VIDEO_EMB_DIR, exist_ok=True)

    print(f"Loading dataset from {DATASET_PATH}...")
    if DATASET_PATH.endswith('.json'):
        ds = load_dataset("json", data_files=DATASET_PATH, split="test")
    else:
        ds = load_dataset(DATASET_PATH, split="test")

    all_video_ids = [item.get('videoID') or item.get('video_id') for item in ds]
    unique_video_ids = sorted(list(set(filter(None, all_video_ids))), key=natural_key)

    if args.subset_file:
        with open(args.subset_file, 'r', encoding='utf-8') as f:
            whitelist = set(json.load(f)['video_ids'])
        before = len(unique_video_ids)
        unique_video_ids = [v for v in unique_video_ids if v in whitelist]
        print(f"[Subset] Applied whitelist {args.subset_file}: {before} -> {len(unique_video_ids)} videos")

    my_assigned_videos = unique_video_ids[rank::world_size]
    my_assigned_videos_set = set(my_assigned_videos)

    print(f"[Node {rank}] Assigned {len(my_assigned_videos)} unique videos.")

    # ----------------------------------------------------
    # Stage 1: Compute embeddings (reverse-accumulation streaming)
    # ----------------------------------------------------
    if RUN_MODE in ['compute_emb', 'all']:
        print(">>> Starting Stage 1: RevAccum Video Embedding Pre-computation")

        manager = mp.Manager()
        queue = manager.Queue()

        processes = []
        stage1_world_size = GPUS_TO_USE * args.stage1_procs_per_gpu
        for worker_rank in range(stage1_world_size):
            p = mp.Process(
                target=worker_precompute_embeddings,
                args=(worker_rank, stage1_world_size, queue, my_assigned_videos,
                      args.model_path, args.chunk_size, args.video_embeddings_path, args.beta)
            )
            p.start()
            processes.append(p)

        pbar = tqdm(total=len(my_assigned_videos), desc=f"Node {rank} Stage 1", unit="video")

        while any(p.is_alive() for p in processes):
            while not queue.empty():
                queue.get()
                pbar.update(1)
            time.sleep(0.1)

        while not queue.empty():
            queue.get()
            pbar.update(1)

        pbar.close()
        for p in processes:
            p.join()
        print(">>> Stage 1 Completed.")

    # ----------------------------------------------------
    # Stage 2: Query-based frame selection
    # ----------------------------------------------------
    if RUN_MODE in ['select_frames', 'all']:
        print(">>> Starting Stage 2: Frame Selection & Saving")

        my_dataset_indices = []
        for idx, vid in enumerate(all_video_ids):
            if vid in my_assigned_videos_set:
                my_dataset_indices.append(idx)

        print(f"[Node {rank}] Found {len(my_dataset_indices)} queries related to my videos.")

        manager = mp.Manager()
        queue = manager.Queue()

        processes = []
        for gpu_rank in range(GPUS_TO_USE):
            p = mp.Process(
                target=worker_select_frames,
                args=(gpu_rank, GPUS_TO_USE, queue, my_dataset_indices,
                      args.model_path, args.output_frames_path,
                      args.num_uniform, args.num_smart, args.video_embeddings_path,
                      args.nms_ratio)
            )
            p.start()
            processes.append(p)

        pbar = tqdm(total=len(my_dataset_indices), desc=f"Node {rank} Stage 2", unit="query")

        while any(p.is_alive() for p in processes):
            while not queue.empty():
                queue.get()
                pbar.update(1)
            time.sleep(0.1)

        while not queue.empty():
            queue.get()
            pbar.update(1)

        pbar.close()
        for p in processes:
            p.join()
        print(">>> Stage 2 Completed.")


if __name__ == "__main__":
    main()
