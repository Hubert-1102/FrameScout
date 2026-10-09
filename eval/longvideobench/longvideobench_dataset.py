from torch.utils.data import Dataset
import os
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
import json

def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(':')
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

# --- 新增：读取离线非均匀采样帧的函数 ---
def load_pre_extracted_frames(folder_path, txt_path):
    """
    假设 txt_path 的格式为每一行: filename timestamp
    例如: 
    frame_0001.jpg 2.50
    frame_0002.jpg 5.33
    """
    frames = []
    frame_timestamps = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 支持空格或逗号分隔
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                img_name = parts[0]
                timestamp = float(parts[1])
                
                img_path = os.path.join(folder_path, img_name)
                # 读取图片并转换为 RGB，跳过损坏/缺失的帧
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: failed to load frame {img_path}: {e}")
                    continue
                
                frames.append(img)
                frame_timestamps.append(timestamp)
                
    # 按时间戳正序排序，防止插入字幕时时序错乱
    sorted_data = sorted(zip(frame_timestamps, frames), key=lambda x: x[0])
    if sorted_data:
        frame_timestamps, frames = zip(*sorted_data)
        return list(frames), list(frame_timestamps)
    else:
        return [], []

def insert_subtitles(subtitles):
    interleaved_list = []
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]
        interleaved_list.append(subtitle_text)
    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            if not isinstance(end, float):
                end = duration
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
                
        if covering_frames:
            interleaved_list.append(subtitle_text)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        interleaved_list.append(frame)
        
    return interleaved_list
    
class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                 pre_extracted_dir=None,  # --- 新增：离线帧的主目录 ---
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text
        self.pre_extracted_dir = pre_extracted_dir

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        self.max_num_frames = max_num_frames
        
    def __getitem__(self, index):
        di = self.data[index]
        inputs = []
        
        if self.max_num_frames == 0:
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
            
        if self.max_num_frames == -1:
            with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
                subtitles = json.load(f)
            inputs = insert_subtitles(subtitles)
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
            
        # --- 核心修改：判断是否走离线帧读取逻辑 ---
        if self.pre_extracted_dir is not None:
            folder_path = os.path.join(self.pre_extracted_dir, di["id"])
            txt_path = os.path.join(folder_path, "timestamps.txt")
            if not os.path.isdir(folder_path) or not os.path.isfile(txt_path):
                print(f"Warning: pre-extracted frames missing for {di['id']} "
                      f"(folder={os.path.exists(folder_path)}, txt={os.path.exists(txt_path)}), "
                      f"falling back to empty frames")
                frames, frame_timestamps = [], []
            else:
                try:
                    frames, frame_timestamps = load_pre_extracted_frames(folder_path, txt_path)
                except Exception as e:
                    print(f"Warning: failed to load pre-extracted frames for {di['id']}: {e}")
                    frames, frame_timestamps = [], []
        else:
            # 兼容原有的 decord 在线均匀采样逻辑
            frames, frame_timestamps = load_video(os.path.join(self.data_path, "videos", di["video_path"]), di["duration"], max_num_frames=self.max_num_frames)
            
        with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
            subtitles = json.load(f)
            
        if self.insert_text:
            inputs = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])
        else:
            inputs = frames

        inputs += ["Question: " + di["question"]]
        inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
        inputs += ["Answer with the option's letter from the given choices directly."]

        return {"inputs": inputs, "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"]}
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]