# FrameScout: Scouting Query-Relevant Frames for Long Video Understanding

Official code for the NeurIPS 2026 paper *FrameScout: Scouting Query-Relevant Frames for Long Video Understanding*.

FrameScout is a lightweight, plug-in keyframe selector for long video understanding. It encodes a video **once** into
query-agnostic, temporally-contextualized frame embeddings, retrieves the most query-relevant frames for every
question by cosine similarity, and feeds only those frames to an **unchanged** downstream VideoLLM.

- **Streaming selector architecture**: a sliding-window KV cache propagates past context across chunks, and
  successor aggregation (`z_i = h_i + β z_{i+1}`) adds future context, under constant KV-cache memory.
- **Frame-query contrastive objective**: aligns frame and query embeddings in a shared space using
  teacher-attention pseudo-labels.

## Repository structure

```
train/
  train_script_two_stage_reverse_accum.py   # contrastive training of the selector
  utils.py                                  # dataset, InternVLWithHead (backbone + projectors)
inference/
  frame_selector_videomme_revaccum.py       # streaming encoding + per-query selection, VideoMME
  frame_selector_longvb_revaccum.py         # LongVideoBench
  frame_selector_mlvu_revaccum.py           # MLVU
  selector_utils.py                         # model loading, successor aggregation
  attention_patch.py                        # RoPE-shifted attention for the sliding-window KV cache
  utils.py                                  # projector definitions
  local_internvl/                           # InternVL2.5 model code with compatibility fixes
eval/
  videomme/eval_videoMME_qwen_selected.py   # downstream QA with Qwen2.5-VL-7B
  longvideobench/qwen2_5_longvideobench.py
  mlvu/eval_mlvu_qwen_selected.py
```

## Installation

Training and frame selection, and the downstream evaluation with Qwen2.5-VL, use different `transformers`
versions, so we recommend two environments.

```bash
# Selector (training + frame selection)
conda create -n framescout python=3.10 -y && conda activate framescout
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional

# Downstream evaluation
conda create -n framescout-eval python=3.10 -y && conda activate framescout-eval
pip install -r eval/requirements.txt
```

## Data preparation

- **Training data**: a JSONL file of teacher-attention pseudo-labels on LLaVA-Video. Each line contains a query,
  100 uniformly sampled frame paths, and per-frame scores/labels:
  ```json
  {"query": "...", "video_id": "...", "image_paths": ["frame_000.jpg", "..."], "labels": [1, 0, "..."], "scores": [0.019, "..."]}
  ```
  Pseudo-labels are obtained from the query-to-visual attention of a teacher VideoLLM (InternVL2.5-8B), following FlexSelect.
- **Benchmarks**: [VideoMME](https://video-mme.github.io), [LongVideoBench](https://longvideobench.github.io), and
  [MLVU](https://github.com/JUNJIE99/MLVU). For each video, pre-extract 1024 uniformly sampled frames into
  `<frames_root>/<video_id>/`.

Dataset locations are set via environment variables:

```bash
export VIDEOMME_DATA=/path/to/videomme            # VideoMME parquet annotations
export VIDEOMME_FRAMES=/path/to/videomme/frames1024
export LONGVB_ANNOTATION=/path/to/LongVideoBench/lvb_val.json
export LONGVB_FRAMES=/path/to/LongVideoBench/frames1024
export MLVU_JSON_DIR=/path/to/MLVU/json
export MLVU_FRAMES=/path/to/MLVU/frames1024
```

## Training

```bash
cd train
torchrun --nproc_per_node=8 train_script_two_stage_reverse_accum.py \
  --model_path OpenGVLab/InternVL2_5-1B \
  --json_path /path/to/pseudo_labels.jsonl \
  --output_dir ./checkpoints \
  --epochs 2 --batch_size 32 --max_num 1 --max_images_per_forward 32 \
  --vlm_lr 5e-5 --proj_lr 5e-4 --projection_dim 1024 --temperature 0.05 --beta 0.3
```

Training logs to Weights & Biases; set `WANDB_MODE=offline` to disable online logging.

## Frame selection

Each selector (1) streams through every video once and caches query-agnostic frame embeddings, and
(2) selects `K=60` frames per question, writing one folder of ordered frames per question.
The default arguments follow the paper setting (`chunk_size=12`, `beta=0.3`, `K=60`).

```bash
cd inference
export FRAMESCOUT_CKPT=/path/to/framescout_checkpoint   # directory containing pytorch_model.bin

python frame_selector_videomme_revaccum.py \
  --model_path $FRAMESCOUT_CKPT \
  --video_embeddings_path ./cache/videomme \
  --output_frames_path ./selected/videomme
```

Use `frame_selector_longvb_revaccum.py` and `frame_selector_mlvu_revaccum.py` in the same way for
LongVideoBench and MLVU. The scripts use 8 GPUs by default (`GPUS_TO_USE`).
If the checkpoint does not record its base model, set `FRAMESCOUT_BASE_MODEL` to the InternVL2.5-1B path.

## Evaluation

```bash
export QWEN_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

cd eval/videomme && python eval_videoMME_qwen_selected.py --frames_path ../../inference/selected/videomme
cd eval/mlvu     && python eval_mlvu_qwen_selected.py     --frames_path ../../inference/selected/mlvu
cd eval/longvideobench && python qwen2_5_longvideobench.py \
  --model_path $QWEN_MODEL --data_path /path/to/LongVideoBench --annotation_file lvb_val.json \
  --frames_path ../../inference/selected/longvb --num_gpus 8
```

