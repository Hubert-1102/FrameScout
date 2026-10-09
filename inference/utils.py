import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random  
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = min(target_ratios, key=lambda x: abs(x[0]/x[1] - aspect_ratio))
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    processed_images = []
    for i in range(best_ratio[1]):
        for j in range(best_ratio[0]):
            box = (j * image_size, i * image_size, (j + 1) * image_size, (i + 1) * image_size)
            processed_images.append(resized_img.crop(box))
    if use_thumbnail:
        thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
        processed_images.append(thumbnail)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    try:
        image = Image.open(image_file).convert('RGB')
    except:
        print('no image')
        image = Image.new('RGB', (448, 448), (128, 128, 128))
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class VideoRetrievalDataset(Dataset):
    def __init__(self, data_source, max_num=12):
        """
        data_source: 可以是 json 路径 (str)，也可以是已经加载好的 list 数据。
        """
        self.data = []
        self.max_num = max_num
        
        if isinstance(data_source, str):
            if os.path.exists(data_source):
                with open(data_source, 'r') as f:
                    for line in f:
                        try:
                            line_data = json.loads(line)
                            self.data.append(line_data)
                        except json.JSONDecodeError:
                            continue # 跳过损坏的行
        elif isinstance(data_source, list):
            self.data = data_source
        else:
            raise ValueError("data_source must be a file path or a list of data items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item['image_paths']
        query = item['query']
        labels = item['labels']
        
        raw_scores = item.get('scores', labels)
        
        pixel_values_list = []
        num_patches_list = []
        
        # 增加简单的错误处理，防止单张坏图导致训练崩溃
        for path in image_paths:
            try:
                # 假设 load_image 是你外部定义的函数
                pv = load_image(path, max_num=self.max_num, input_size=448)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                pv = torch.zeros((1, 3, 448, 448)) 
            
            pixel_values_list.append(pv)
            num_patches_list.append(pv.shape[0])
            
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        return {
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "query": query,
            "labels": torch.tensor(labels, dtype=torch.float), # 保持用于 Hit@1 评估
            "scores": torch.tensor(raw_scores, dtype=torch.float), # 新增: 用于 Rank Loss
            "num_images": len(image_paths)
        }

def custom_collate_fn(batch):
    return batch




# ==========================================
# 模型类 (MLPProjector, InternVLWithHead 保持不变)
# ==========================================
class MLPProjector(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        hidden_size = output_size 
        # hidden_size = input_size 
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_size, output_size, bias=False)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class InternVLWithHead(nn.Module):
    def __init__(self, original_model, input_size, output_size):
        super().__init__()
        self.model = original_model
        self.img_projector = MLPProjector(input_size,output_size)
        self.txt_projector = MLPProjector(input_size,output_size)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)




