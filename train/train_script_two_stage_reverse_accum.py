import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
import random
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
from utils import VideoRetrievalDataset, custom_collate_fn, InternVLWithHead
import wandb


# ==========================================
# Loss (unchanged, same as the original single-direction script)
# ==========================================
class TopBottomContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embedding, labels):
        image_norm = F.normalize(image_embeddings, p=2, dim=1)
        text_norm = F.normalize(text_embedding, p=2, dim=1)
        scores = torch.mv(image_norm, text_norm.squeeze(0))
        if dist.get_rank() == 0 and random.random() < 0.5:
            print(f"\n[DEBUG] Score Mean: {scores.mean().item():.4f}, "
                  f"Max: {scores.max().item():.4f}, Min: {scores.min().item():.4f}")
            print(f"[DEBUG] Num Candidates (N): {len(scores)}")
        logits = scores / self.temperature

        pos_indices = torch.nonzero(labels > 0.5).squeeze(1)
        neg_indices = torch.nonzero(labels < 0.5).squeeze(1)

        if len(pos_indices) == 0:
            return scores.sum() * 0.0

        if len(neg_indices) > 0:
            exp_neg_sum = torch.sum(torch.exp(logits[neg_indices]))
        else:
            exp_neg_sum = torch.tensor(0.0, device=scores.device)

        loss = 0.0
        for pos_idx in pos_indices:
            exp_pos = torch.exp(logits[pos_idx])
            denominator = exp_pos + exp_neg_sum
            log_prob = torch.log(exp_pos / (denominator + 1e-8))
            loss += log_prob

        loss = -loss / len(pos_indices)
        return loss


# ==========================================
# Standard chunked image-embedding extraction (forward order only).
# Identical to train_script_two_stage.py.
# ==========================================
def get_image_embeddings_in_chunks(model, tokenizer, pixel_values, num_patches_list,
                                   chunk_size, prompt_configs):
    (SYSTEM, USER, ASSISTANT, IMG_START, IMG_CONTEXT, IMG_END, _) = prompt_configs
    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END)
    all_embeddings = []
    total_images = len(num_patches_list)
    cumulative_patches = [0]
    for n in num_patches_list:
        cumulative_patches.append(cumulative_patches[-1] + n)

    raw_model = model.module if hasattr(model, "module") else model
    device = pixel_values.device

    for i in range(0, total_images, chunk_size):
        end_idx = min(i + chunk_size, total_images)
        current_patches_list = num_patches_list[i:end_idx]
        start_patch_idx = cumulative_patches[i]
        end_patch_idx = cumulative_patches[end_idx]
        chunk_pixel_values = pixel_values[start_patch_idx:end_patch_idx].to(device)

        chunk_imgs_str = ""
        for num_patches in current_patches_list:
            img_tokens = IMG_START + IMG_CONTEXT * raw_model.num_image_token * num_patches + IMG_END
            chunk_imgs_str += img_tokens + "\n"

        full_chunk_prompt = f"{SYSTEM}{USER}{chunk_imgs_str}{ASSISTANT}"
        chunk_inputs = tokenizer(full_chunk_prompt, return_tensors='pt', padding=True)
        chunk_input_ids = chunk_inputs['input_ids'].to(device)
        chunk_attention_mask = chunk_inputs['attention_mask'].to(device)
        image_flags = torch.ones((chunk_pixel_values.shape[0], 1), dtype=torch.long).to(device)

        outputs = raw_model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
            pixel_values=chunk_pixel_values,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        last_hidden_state = outputs.hidden_states[-1]
        end_indices = torch.nonzero(chunk_input_ids[0] == img_end_token_id, as_tuple=True)[0]
        if len(end_indices) != len(current_patches_list):
            continue
        chunk_embeddings = last_hidden_state[0, end_indices, :]
        all_embeddings.append(chunk_embeddings)

    if len(all_embeddings) > 0:
        return torch.cat(all_embeddings, dim=0)
    return None


# ==========================================
# Reverse Accumulation:
#   Given per-frame hidden states h of shape (N, H),
#   produce h'[i] = h[i] + beta * h'[i+1]
#   (from the last frame to the first)
#
# Equivalent closed form:
#   h'[i] = sum_{k=i..N-1} beta^(k - i) * h[k]
#
# This gives every frame a "future view" via an exponentially decaying sum
# of all later frames, which is the "reverse field of view" we want.
# Implementation uses an in-order Python loop over frames (N is small, e.g. <= a few hundred),
# keeping full autograd through h.
# ==========================================
def reverse_accumulate(h, beta):
    """
    h:     Tensor (N, H)
    beta:  float
    returns: Tensor (N, H) with the reverse-cumulative mixing applied.
    """
    if h is None or h.shape[0] == 0:
        return h

    # Build a list so each h'[i] keeps a fresh Tensor in the autograd graph
    n = h.shape[0]
    out = [None] * n
    out[n - 1] = h[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = h[i] + beta * out[i + 1]
    return torch.stack(out, dim=0)


def forward_step(model, tokenizer, sample, device, criterion, prompt_configs,
                 max_images_per_forward, beta=0.5, return_acc=False):
    """
    Training-time forward WITHOUT streaming:
      1) Chunked forward pass -> h (N, H)
      2) Reverse accumulation: h'[i] = h[i] + beta * h'[i+1]
      3) Normalize h'
      4) Project with the ORIGINAL single-direction MLP (hidden_size -> projection_dim)
    """
    pixel_values = sample['pixel_values'].to(device, torch.bfloat16)
    num_patches_list = sample['num_patches_list']
    query_text = sample['query']
    labels = sample['labels'].to(device)

    wrapper = model.module if hasattr(model, 'module') else model
    inner_model = wrapper.model

    # ---- 1) Image Branch (forward order, chunked) ----
    h_fwd = get_image_embeddings_in_chunks(
        inner_model, tokenizer, pixel_values, num_patches_list,
        max_images_per_forward, prompt_configs
    )
    if h_fwd is None:
        return None if not return_acc else (None, 0)

    # ---- 2) Reverse accumulation (in-place along the frame axis) ----
    h_rev_accum = reverse_accumulate(h_fwd, beta)

    # ---- 3) Normalize + project (same signature as original single-direction path) ----
    image_embeddings = F.normalize(h_rev_accum, dim=-1)
    image_features = wrapper.img_projector(image_embeddings)

    # ---- 4) Text Branch (unchanged) ----
    text_template = prompt_configs[-1]
    full_text_prompt = text_template.format(query_text)
    text_inputs = tokenizer(full_text_prompt, return_tensors='pt')
    text_input_ids = text_inputs['input_ids'].to(device)
    text_attention_mask = text_inputs['attention_mask'].to(device)

    raw_model = inner_model.module if hasattr(inner_model, 'module') else inner_model
    text_outputs = raw_model.language_model(
        input_ids=text_input_ids,
        attention_mask=text_attention_mask,
        output_hidden_states=True,
        use_cache=False
    )
    text_embedding = text_outputs.hidden_states[-1][:, -1, :]
    text_embedding = F.normalize(text_embedding, dim=-1)
    text_features = wrapper.txt_projector(text_embedding)

    if return_acc:
        image_norm = F.normalize(image_features, p=2, dim=1)
        text_norm = F.normalize(text_features, p=2, dim=1)
        scores = torch.mv(image_norm, text_norm.squeeze(0))
        best_idx = torch.argmax(scores)
        is_correct = (labels[best_idx] == 1).float().item()
        return is_correct
    else:
        loss = criterion(image_features, text_features, labels)
        return loss


# ==========================================
# Evaluation (same interface as the original script, with beta passthrough)
# ==========================================
@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, prompt_configs, max_images_per_forward, beta):
    model.eval()
    total_hits = 0.0
    total_samples = 0.0

    if dist.get_rank() == 0:
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    else:
        pbar = dataloader

    for batch in pbar:
        for sample in batch:
            is_correct = forward_step(
                model, tokenizer, sample, device, None,
                prompt_configs, max_images_per_forward,
                beta=beta, return_acc=True
            )
            if is_correct is not None:
                total_hits += is_correct
                total_samples += 1

    stats = torch.tensor([total_hits, total_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_hits = stats[0].item()
    global_total = stats[1].item()

    accuracy = global_hits / global_total if global_total > 0 else 0.0
    model.train()
    return accuracy


import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="InternVL2.5 Video Retrieval Training (Reverse-Accumulation Bidirectional)"
    )
    parser.add_argument("--model_path", type=str,
                        default="OpenGVLab/InternVL2_5-1B")
    parser.add_argument("--json_path", type=str,
                        required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vlm_lr", type=float, default=5e-5)
    parser.add_argument("--proj_lr", type=float, default=5e-5)
    parser.add_argument("--max_num", type=int, default=1)
    parser.add_argument("--max_images_per_forward", type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--projection_dim", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)

    # Reverse accumulation coefficient: h'[i] = h[i] + beta * h'[i+1]
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Reverse accumulation decay factor, h'[i] = h[i] + beta * h'[i+1]")

    args = parser.parse_args()
    return args


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    global_rank = dist.get_rank()

    MODEL_PATH = args.model_path
    JSON_PATH = args.json_path
    CHECKPOINT_PATH = args.checkpoint_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MAX_IMAGES_PER_FORWARD = args.max_images_per_forward
    PROJECTION_DIM = args.projection_dim
    MAX_NUM = args.max_num
    EVAL_INTERVAL = args.eval_interval
    VLM_LR = args.vlm_lr
    PROJ_LR = args.proj_lr
    TEMPERATURE = args.temperature
    BETA = args.beta

    if global_rank == 0:
        print("=" * 60)
        print("Training Configuration (Reverse-Accumulation Bidirectional)")
        print(f"Model Path: {MODEL_PATH}")
        print(f"Dataset:    {JSON_PATH}")
        print(f"Beta:       {BETA}")
        print("=" * 60)
        wandb.init(config=args)

    # 1. Load data
    all_data = []
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
    else:
        if global_rank == 0:
            print("Dataset file not found!")
        return

    random.seed(42)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * 0.95)
    train_data_list = all_data[:split_idx]
    val_data_list = all_data[split_idx:]

    if global_rank == 0:
        print(f"Total Data: {len(all_data)}")
        print(f"Train Size: {len(train_data_list)}")
        print(f"Val Size:   {len(val_data_list)}")

    # 2. Model init
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True
    ).to(device)

    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = True
    if hasattr(model, 'mlp1'):
        for param in model.mlp1.parameters():
            param.requires_grad = True

    model.language_model.gradient_checkpointing_disable()
    model.vision_model.gradient_checkpointing_disable()

    hidden_size = model.language_model.config.hidden_size
    # IMPORTANT: reuse the ORIGINAL single-direction wrapper.
    # img_projector: hidden_size -> projection_dim (NOT 2*hidden_size).
    model = InternVLWithHead(model, hidden_size, PROJECTION_DIM).to(device, dtype=torch.bfloat16)

    # Because MLP shapes are identical to the original single-direction training,
    # stage1 / stage2 single-direction checkpoints can be loaded directly.
    if CHECKPOINT_PATH:
        if global_rank == 0:
            print(f"加载权重: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    else:
        if global_rank == 0:
            print('未加载权重')

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Prompt Setup
    raw_model = model.module.model
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    raw_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    PROMPT_CONFIGS = (
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n",
        "<|im_start|>user\n",
        "\n<|im_end|>\n<|im_start|>assistant\n",
        '<img>', IMG_CONTEXT_TOKEN, '</img>',
        "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
    )

    # 3. Dataset & Dataloader
    train_dataset = VideoRetrievalDataset(train_data_list, max_num=MAX_NUM)
    val_dataset = VideoRetrievalDataset(val_data_list, max_num=MAX_NUM)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, num_workers=8, pin_memory=False,
                                  batch_size=BATCH_SIZE, collate_fn=custom_collate_fn,
                                  sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                collate_fn=custom_collate_fn, sampler=val_sampler)

    # Parameter groups
    head_params = []
    base_params = []
    head_names = ['img_projector', 'txt_projector']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {'params': base_params, 'lr': VLM_LR},
        {'params': head_params, 'lr': PROJ_LR}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    criterion = TopBottomContrastiveLoss(temperature=TEMPERATURE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataloader) * EPOCHS * 0.05),
        num_training_steps=len(train_dataloader) * EPOCHS
    )

    if global_rank == 0:
        print(f"Starting training loop (reverse-accumulation, beta={BETA})...")

    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        if global_rank == 0:
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", ncols=120)
        else:
            pbar = train_dataloader

        for step, batch_data in enumerate(pbar):
            accum_loss = 0.0
            optimizer.zero_grad()

            with model.no_sync():
                for i, sample in enumerate(batch_data):
                    loss = forward_step(model, tokenizer, sample, device, criterion,
                                        PROMPT_CONFIGS, MAX_IMAGES_PER_FORWARD, beta=BETA)
                    if loss is not None:
                        loss = loss / len(batch_data)
                        loss.backward()
                        accum_loss += loss.item()

            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if global_rank == 0:
                wandb.log({"train/loss": accum_loss,
                           "train/lr": optimizer.param_groups[0]['lr']}, step=global_step)
                pbar.set_postfix({"Loss": f"{accum_loss:.4f}"})

            global_step += 1

            if (global_step) % EVAL_INTERVAL == 0:
                if global_rank == 0:
                    print(f"\n[Step {global_step}] Running evaluation...")
                val_acc = evaluate(model, tokenizer, val_dataloader, device,
                                   PROMPT_CONFIGS, MAX_IMAGES_PER_FORWARD, beta=BETA)
                if global_rank == 0:
                    print(f"*** [Step {global_step}] Validation Accuracy (Hit@1): {val_acc:.4f} ***")
                    wandb.log({"val/accuracy": val_acc}, step=global_step)

            if step % 10 == 0:
                torch.cuda.empty_cache()

        if global_rank == 0:
            save_path = os.path.join(
                args.output_dir,
                f"stage2_revaccum_beta={BETA}_image={MAX_IMAGES_PER_FORWARD}_maxnum={MAX_NUM}_"
                f"batchsize={BATCH_SIZE}_vlmlr={VLM_LR}_projlr={PROJ_LR}_"
                f"projdim={PROJECTION_DIM}_{current_time}",
                f"epoch_{epoch+1}"
            )
            os.makedirs(save_path, exist_ok=True)
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(model.module.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            print(f"Checkpoint saved to: {save_path}")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
