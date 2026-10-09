import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import torch.multiprocessing as mp
import hashlib
import argparse

def natural_key(string_):
    """
    将字符串中的数字部分转换为整数，实现自然排序。
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

parser = argparse.ArgumentParser()
# ================= 配置区域 =================
# 模型路径
MODEL_PATH = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct") 

# 数据集本地路径
DATASET_PATH = os.environ.get("VIDEOMME_DATA", "/path/to/videomme") 
parser.add_argument("--frames_path", type=str, required=True, help="Root path for saving selected frames")
parser.add_argument("--subset_file", type=str, default=None,
                    help="Optional JSON whitelist ({'video_ids': [...]}); if set, only questions "
                         "whose videoID is in the whitelist are evaluated (avoids NoFrames dilution).")
args = parser.parse_args()
# 这里指向上一轮代码输出的文件夹路径
FRAMES_ROOT_PATH = args.frames_path

# 使用的 GPU 数量
NUM_GPUS = 8 
# ===========================================

def get_duration_category(duration):
    """解析 duration 类别，兼容字符串和数值（秒）格式"""
    if isinstance(duration, str):
        d_lower = duration.lower()
        if 'short' in d_lower: return 'short'
        if 'medium' in d_lower: return 'medium'
        if 'long' in d_lower: return 'long'
        return 'unknown'
    elif isinstance(duration, (int, float)):
        # Video-MME 官方标准: short < 2min, medium 2-15min, long > 15min
        if duration < 120: return 'short'
        elif duration <= 900: return 'medium'
        else: return 'long'
    return 'unknown'

def get_frame_paths(video_id, question, root_path):
    """根据 video_id 和 question 找到对应的精选帧文件夹"""
    q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()[:8]
    safe_video_id = video_id.replace('/', '_').replace('\\', '_')
    folder_name = f"{safe_video_id}_{q_hash}"
    
    frame_dir = os.path.join(root_path, folder_name)
    
    if not os.path.exists(frame_dir):
        print(f"Missing folder: {frame_dir}")
        return []

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    try:
        all_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(valid_exts)], key=natural_key)
    except FileNotFoundError:
        return []
    
    if not all_files:
        return []

    frame_uris = []
    for f in all_files:
        abs_path = os.path.abspath(os.path.join(frame_dir, f))
        frame_uris.append(f"file://{abs_path}")
        
    return frame_uris

def format_options(options):
    return "\n".join(options)

def extract_answer(output_text):
    """提取答案"""
    text = output_text.strip()
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Option\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    if text and text[0].upper() in ['A', 'B', 'C', 'D']: return text[0].upper()
    match = re.search(r"answer is\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "Unknown"

def eval_worker(rank, subset_indices, result_queue):
    """工作进程函数：运行在单个 GPU 上"""
    try:
        device = f"cuda:{rank}"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device}, 
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        if DATASET_PATH.endswith('.json'):
            full_dataset = load_dataset("json", data_files=DATASET_PATH, split="test")
        else:
            full_dataset = load_dataset(DATASET_PATH, split="test")
            
        dataset_shard = full_dataset.select(subset_indices)
        
        for item in dataset_shard:
            video_id = item.get('videoID') or item.get('video_id')
            question = item['question']
            options = item['options']
            ground_truth = item.get('answer', None)
            
            # 提取 duration 字段
            duration_raw = item.get('duration', 'unknown')
            
            frame_uris = get_frame_paths(video_id, question, FRAMES_ROOT_PATH)
            
            if not frame_uris:
                print('error: NoFrames')
                result_queue.put((False, duration_raw, video_id, "NoFrames", ground_truth))
                continue

            options_str = format_options(options)
            prompt_text = (
                f"Question: {question}\n"
                f"Options:\n{options_str}\n"
                "Answer with the option letter directly."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frame_uris,
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample = False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_answer = extract_answer(output_text)
            
            is_correct = (pred_answer == ground_truth) if ground_truth else False
            
            # 将 duration_raw 加入队列
            result_queue.put((is_correct, duration_raw, video_id, pred_answer, ground_truth))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Rank {rank} Error: {e}")
    finally:
        result_queue.put(None)

def main():
    mp.set_start_method('spawn', force=True)

    print(f"正在加载数据集索引以进行切分: {DATASET_PATH} ...")
    if DATASET_PATH.endswith('.json'):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="test")
    else:
        dataset = load_dataset(DATASET_PATH, split="test")
        
    total_samples = len(dataset)
    indices = np.arange(total_samples)

    if args.subset_file:
        import json as _json
        with open(args.subset_file, 'r', encoding='utf-8') as f:
            whitelist = set(_json.load(f)['video_ids'])
        all_vids = dataset['videoID'] if 'videoID' in dataset.column_names else dataset['video_id']
        indices = np.array([i for i in range(total_samples) if all_vids[i] in whitelist])
        print(f"[Subset] Applied whitelist {args.subset_file}: {total_samples} -> {len(indices)} questions "
              f"({len(whitelist)} whitelisted videos)")

    print(f"总样本数: {total_samples}, 评估样本数: {len(indices)}, GPU 数量: {NUM_GPUS}")

    split_indices = np.array_split(indices, NUM_GPUS)

    result_queue = mp.Queue()
    
    processes = []
    print("启动工作进程...")
    for rank in range(NUM_GPUS):
        p = mp.Process(target=eval_worker, args=(rank, split_indices[rank], result_queue))
        p.start()
        processes.append(p)
    
    finished_workers = 0
    correct_count = 0
    processed_count = 0
    
    # 时长统计字典
    stats = {
        'short': {'correct': 0, 'total': 0},
        'medium': {'correct': 0, 'total': 0},
        'long': {'correct': 0, 'total': 0},
        'unknown': {'correct': 0, 'total': 0}
    }
    
    pbar = tqdm(total=len(indices), desc="Evaluated")
    results_log = []

    while finished_workers < NUM_GPUS:
        item = result_queue.get()
        
        if item is None:
            finished_workers += 1
            continue
            
        # 解包新增的 duration_raw
        is_correct, duration_raw, video_id, pred, gt = item
        
        # 分类统计
        cat = get_duration_category(duration_raw)
        stats[cat]['total'] += 1
        if is_correct:
            stats[cat]['correct'] += 1
            correct_count += 1
            
        processed_count += 1
        
        # 将时长的原始值和分类都写进日志
        results_log.append({
            "video_id": video_id, 
            "duration": duration_raw,
            "duration_category": cat,
            "pred": pred, 
            "gt": gt, 
            "correct": is_correct
        })

        current_acc = (correct_count / processed_count) * 100
        pbar.set_description(f"Acc: {current_acc:.2f}% ({correct_count}/{processed_count})")
        pbar.update(1)

    pbar.close()
    
    print("等待所有进程完全退出...")
    for p in processes:
        p.join()

    print(f"\n========================================")
    print(f"Final Evaluation Result ({NUM_GPUS} GPUs)")
    print(f"========================================")
    print(f"Total Questions: {processed_count}")
    print(f"Overall Accuracy: {correct_count / processed_count * 100:.2f}%" if processed_count > 0 else "Overall Accuracy: 0%")
    print(f"----------------------------------------")
    
    for cat in ['short', 'medium', 'long', 'unknown']:
        total = stats[cat]['total']
        if total > 0:
            correct = stats[cat]['correct']
            acc = (correct / total) * 100
            print(f"[{cat.capitalize():<7}] Accuracy: {acc:5.2f}% ({correct}/{total})")

    print(f"========================================")
    
    import json
    with open("evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()