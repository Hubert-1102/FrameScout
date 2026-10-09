import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import torch.multiprocessing as mp
from PIL import Image
import argparse

from longvideobench_dataset import LongVideoBenchDataset

def extract_answer(output_text):
    """提取答案 (适配 A-E 选项)"""
    text = output_text.strip()
    # 匹配 Answer: A 或 Option A
    match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Option\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 匹配首字母
    if text and text[0].upper() in ['A', 'B', 'C', 'D', 'E']: 
        return text[0].upper()
        
    # 匹配 answer is A
    match = re.search(r"answer is\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    return "Unknown"

def eval_worker(rank, subset_indices, result_queue, args):
    """
    工作进程函数：运行在单个 GPU 上
    """
    try:
        device = f"cuda:{rank}"
        
        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device}, 
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        # 在 Worker 内部实例化 Dataset，防止多进程共享文件描述符产生死锁或竞态
        dataset = LongVideoBenchDataset(
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            max_num_frames=args.max_num_frames,
            insert_text=True,
            insert_frame=True,
            pre_extracted_dir=args.frames_path  # 传入预提取帧目录
        )
        
        for idx in subset_indices:
            data = dataset[idx]
            video_id = data["id"]
            inputs = data["inputs"]
            ground_truth = data["correct_choice"]
            
            # 将 Dataset 吐出的 [Image, str, Image, str...] 转换为 Qwen 接受的 content 格式
            content = []
            for item in inputs:
                if isinstance(item, Image.Image):
                    content.append({"type": "image", "image": item})
                elif isinstance(item, str):
                    # 补充换行符让文本排版更清晰
                    content.append({"type": "text", "text": item + "\n"})
                else:
                    pass

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            model_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(device)

            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "num_beams": args.num_beams,
                }
                if args.do_sample:
                    generate_kwargs.update({
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    })
                    if args.top_k is not None:
                        generate_kwargs["top_k"] = args.top_k

                generated_ids = model.generate(**model_inputs, **generate_kwargs)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_answer = extract_answer(output_text)
            is_correct = (pred_answer == ground_truth)
            
            result_queue.put((is_correct, video_id, pred_answer, ground_truth))
            
    except Exception as e:
        import traceback
        print(f"Rank {rank} Error: {e}")
        traceback.print_exc()
    finally:
        result_queue.put(None)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on LongVideoBench")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model directory")
    parser.add_argument("--data_path", type=str, default="/path/to/LongVideoBench", help="Path to the dataset directory")
    parser.add_argument("--annotation_file", type=str, default="lvb_val.json", help="Annotation JSON file name")
    parser.add_argument("--max_num_frames", type=int, default=60, help="Maximum number of frames (fallback if pre_extracted_dir is None)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use for evaluation")
    parser.add_argument("--frames_path", type=str, default=None, help="Path to the directory containing pre-extracted non-uniform frames and timestamps")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of newly generated tokens")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling during generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature, only used when --do_sample is set")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p, only used when --do_sample is set")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling, only used when --do_sample is set")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--subset_file", type=str, default=None,
                        help="Optional JSON whitelist ({'video_ids': [...]}); if set, only questions "
                             "whose video_id is in the whitelist are evaluated.")
    return parser.parse_args()

def main():
    args = parse_args()
    mp.set_start_method('spawn', force=True)

    print("正在加载数据集以获取总样本数...")
    # 这里实例化只为了获取数据集长度。
    # 设置 max_num_frames=0 可以跳过解码视频的耗时操作，加快主进程初始化速度。
    dummy_dataset = LongVideoBenchDataset(
        data_path=args.data_path,
        annotation_file=args.annotation_file,
        max_num_frames=0,
        pre_extracted_dir=args.frames_path
    )
    total_samples = len(dummy_dataset)
    indices = np.arange(total_samples)

    if args.subset_file:
        import json as _json
        with open(args.subset_file, 'r', encoding='utf-8') as f:
            whitelist = set(_json.load(f)['video_ids'])
        indices = np.array([i for i in range(total_samples)
                            if dummy_dataset.data[i].get('video_id') in whitelist])
        print(f"[Subset] Applied whitelist {args.subset_file}: {total_samples} -> {len(indices)} questions "
              f"({len(whitelist)} whitelisted videos)")

    print(f"总样本数: {total_samples}, 评估样本数: {len(indices)}, GPU 数量: {args.num_gpus}")
    print(
        "Generation config: "
        f"max_new_tokens={args.max_new_tokens}, "
        f"do_sample={args.do_sample}, "
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"top_k={args.top_k}, "
        f"num_beams={args.num_beams}"
    )
    
    # 将索引切分给各个 GPU
    split_indices = np.array_split(indices, args.num_gpus)

    result_queue = mp.Queue()
    processes = []
    
    print("启动工作进程...")
    for rank in range(args.num_gpus):
        # 将 args 传递给 eval_worker
        p = mp.Process(target=eval_worker, args=(rank, split_indices[rank], result_queue, args))
        p.start()
        processes.append(p)

    finished_workers = 0
    correct_count = 0
    processed_count = 0
    
    pbar = tqdm(total=len(indices), desc="Evaluated")
    
    while finished_workers < args.num_gpus:
        item = result_queue.get() 
        
        if item is None:
            finished_workers += 1
            continue
            
        is_correct, video_id, pred, gt = item
        
        processed_count += 1
        if is_correct:
            correct_count += 1
            
        current_acc = (correct_count / processed_count) * 100
        pbar.set_description(f"Acc: {current_acc:.2f}% ({correct_count}/{processed_count})")
        pbar.update(1)

    pbar.close()
    
    print("等待所有进程完全退出...")
    for p in processes:
        p.join()

    print(f"\n========================================")
    print(f"Final Evaluation Result ({args.num_gpus} GPUs)")
    print(f"Total Questions: {processed_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {correct_count / processed_count * 100:.2f}%" if processed_count > 0 else "Accuracy: 0%")
    print(f"========================================")

if __name__ == "__main__":
    main()