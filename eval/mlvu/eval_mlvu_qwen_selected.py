import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import torch.multiprocessing as mp
import hashlib 
import argparse
import glob
import json

def natural_key(string_):
    """
    将字符串中的数字部分转换为整数，实现自然排序。
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

parser = argparse.ArgumentParser()
# ================= 配置区域 =================
# 模型路径
MODEL_PATH = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct") 

# MLVU JSON 文件夹路径
DATASET_JSON_DIR = os.environ.get("MLVU_JSON_DIR", "/path/to/MLVU/json") 

# 接收 Bash 脚本传进来的选帧路径
parser.add_argument("--frames_path", type=str, required=True, help="Root path for saving selected frames")
args = parser.parse_args()

FRAMES_ROOT_PATH = args.frames_path
NUM_GPUS = 8 
# Max samples per task (set to None for no limit)
MAX_PER_TASK = None
# ===========================================

def load_mlvu_dataset(json_dir, max_per_task=None):
    """
    加载并合并 MLVU 目录下前 7 个（选择题类型）JSON 文件
    """
    dataset = []
    # 获取目录下所有的 JSON 文件
    all_json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    if not all_json_files:
        print(f"警告：在 {json_dir} 中未找到任何 JSON 文件！")
        return dataset

    # 过滤逻辑：只需要以 1-7 开头的文件
    target_prefixes = ['1', '2', '3', '4', '5', '6', '7']  # Only tasks 1, 2, 3, 4, 5, 6, 7
    

    selected_files = []
    for json_file in all_json_files:
        file_name = os.path.basename(json_file)
        if any(file_name.startswith(p) for p in target_prefixes):
            selected_files.append(json_file)
    
    print(f"已筛选文件: {[os.path.basename(f) for f in selected_files]}")

    for json_file in selected_files:
        category = os.path.basename(json_file).replace('.json', '')
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            print(f"正在加载任务: {category}, 样本数: {len(file_data)}")
            if max_per_task is not None and len(file_data) > max_per_task:
                file_data = file_data[:max_per_task]
                print(f"  -> 限制为前 {max_per_task} 个样本")
            for item in file_data:
                item['category'] = category
                item['video_id'] = os.path.splitext(item['video'])[0] 
                dataset.append(item)
    return dataset

def get_frame_paths(category, video_id, question, root_path):
    """
    根据 category, video_id 和 question 找到对应的精选帧文件夹
    """
    q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()[:8]
    safe_video_id = video_id.replace('/', '_').replace('\\', '_')
    folder_name = f"{safe_video_id}_{q_hash}"
    
    # 加上 category 层级
    frame_dir = os.path.join(root_path, category, folder_name)
    
    if not os.path.exists(frame_dir):
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

def extract_answer(output_text):
    """提取答案"""
    text = output_text.strip()
    match = re.search(r"Answer:\s*([A-Z])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Option\s*([A-Z])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    if text and text[0].upper() in ['A', 'B', 'C', 'D', 'E', 'F']: return text[0].upper()
    match = re.search(r"answer is\s*([A-Z])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "Unknown"

def eval_worker(rank, subset_indices, full_dataset, result_queue):
    """
    工作进程函数：运行在单个 GPU 上
    """
    try:
        device = f"cuda:{rank}"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device}, 
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        for idx in subset_indices:
            item = full_dataset[idx]
            
            video_id = item['video_id']
            category = item['category']
            question = item['question']
            candidates = item['candidates']
            ground_truth_text = item.get('answer', None)
            task_desc = item.get('question_type', category)
            
            # 获取图片 URI
            frame_uris = get_frame_paths(category, video_id, question, FRAMES_ROOT_PATH)
            
            if not frame_uris:
                result_queue.put((False, video_id, category, question, "NoFrames", "Error", ""))
                continue

            # ================= 选项与答案字母映射逻辑 =================
            cand_stripped = [str(c).strip() for c in candidates]
            gt_stripped = str(ground_truth_text).strip() if ground_truth_text else ""
            
            gt_letter = None
            if gt_stripped in cand_stripped:
                gt_letter = chr(ord('A') + cand_stripped.index(gt_stripped))
                
            options_str = ""
            for i, c in enumerate(cand_stripped):
                options_str += f"{chr(ord('A') + i)}. {c}\n"
            # ========================================================

            prompt_text = (
                f"Task: {task_desc}\n"
                f"Question: {question}\n"
                f"Options:\n{options_str.strip()}\n"
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
                generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_answer = extract_answer(output_text)
            
            is_correct = (pred_answer == gt_letter) if gt_letter else False
            
            result_queue.put((is_correct, video_id, category, question, pred_answer, gt_letter, output_text))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Rank {rank} Error: {e}")
    finally:
        result_queue.put(None)

def main():
    mp.set_start_method('spawn', force=True)

    print(f"正在加载 MLVU 数据集...")
    dataset = load_mlvu_dataset(DATASET_JSON_DIR, max_per_task=MAX_PER_TASK)
        
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    
    print(f"总样本数: {total_samples}, GPU 数量: {NUM_GPUS}")
    
    split_indices = np.array_split(indices, NUM_GPUS)

    result_queue = mp.Queue()
    
    processes = []
    print("启动工作进程...")
    for rank in range(NUM_GPUS):
        p = mp.Process(target=eval_worker, args=(rank, split_indices[rank], dataset, result_queue))
        p.start()
        processes.append(p)
    
    finished_workers = 0
    correct_count = 0
    processed_count = 0
    
    # 用于记录细分任务的成绩
    category_stats = {}
    results_log = []

    pbar = tqdm(total=total_samples, desc="Evaluated")

    while finished_workers < NUM_GPUS:
        item = result_queue.get()
        
        if item is None:
            finished_workers += 1
            continue
            
        is_correct, video_id, category, question, pred, gt, raw_output = item
        
        processed_count += 1
        if is_correct:
            correct_count += 1
            
        # 统计细分任务
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1
        
        results_log.append({
            "video_id": video_id, 
            "question": question,
            "category": category,
            "pred": pred, 
            "gt": gt, 
            "correct": is_correct,
            "raw_output": raw_output
        })

        current_acc = (correct_count / processed_count) * 100
        pbar.set_description(f"Acc: {current_acc:.2f}% ({correct_count}/{processed_count})")
        pbar.update(1)

    pbar.close()
    
    print("等待所有进程完全退出...")
    for p in processes:
        p.join()

    # ================= 打印最终报告 =================
    print(f"\n========================================")
    print(f"MLVU Final Evaluation Result ({NUM_GPUS} GPUs)")
    print(f"Total Questions: {processed_count}")
    print(f"Correct Answers: {correct_count}")
    overall_acc = correct_count / processed_count * 100 if processed_count > 0 else 0
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"----------------------------------------")
    print(f"Breakdown by Category:")
    
    # 按字母顺序打印细分成绩
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  - {cat.ljust(18)}: {cat_acc:5.2f}%  ({stats['correct']}/{stats['total']})")
    print(f"========================================\n")
    
    # 保存结果
    log_file_path = os.path.join(FRAMES_ROOT_PATH, "mlvu_evaluation_results.json")
    with open(log_file_path, "w", encoding='utf-8') as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    print(f"详细评估结果已保存至: {log_file_path}")

if __name__ == "__main__":
    main()