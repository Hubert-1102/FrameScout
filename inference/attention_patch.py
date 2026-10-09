import torch
import torch.nn as nn
import math
import types
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, Qwen2Attention

def qwen2_raw_cache_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor], # 这里传进来的通常只包含当前片段的 pos
    attention_mask: torch.Tensor = None,
    past_key_values: object = None,
    use_cache: bool = False,
    cache_position: torch.LongTensor = None,
    **kwargs,
):
    """
    修改版 Forward：
    1. 存入 Cache 的是 RAW Key/Value (无 RoPE)。
    2. 从 Cache 取出完整 RAW KV 后，根据当前窗口推算连续的 position_ids。
    3. 现场调用 self.rotary_emb 计算全量的 cos/sin 并应用。
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 1. Projection (Q, K, V) - 此时都是原始值
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # =========================================================================
    # 核心修改点 1: 在应用 RoPE 之前，先把 原始(Raw) 的 K/V 存入 Cache
    # =========================================================================
    if past_key_values is not None:
        # 这里存入的是没有旋转过的 K 和 V
        # 注意：我们需要确保 MemoryManager 在 prune 的时候处理的是 raw data (这没问题，tensor 操作不关心含义)
        cache_kwargs = {"sin": None, "cos": None, "cache_position": cache_position} # 传 None 防止内部做额外操作
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    
    # 此时 key_states 和 value_states 已经是包含 (Past + Current) 的完整序列了
    # 且它们目前都是 RAW 的 (没有位置信息)

    # =========================================================================
    # 核心修改点 2: 重新计算连续的 Position IDs 并应用 RoPE
    # =========================================================================
    
    # 获取 kwargs 里的 position_ids (这是当前片段的全局位置)
    # 通常 shape 是 [1, seq_len]
    current_position_ids = kwargs.get("position_ids")
    
    if current_position_ids is None:
        # 兜底：如果没有传 position_ids，就默认从 cache 长度推算
        seq_len_total = key_states.shape[2]
        current_position_ids = torch.arange(seq_len_total - input_shape[1], seq_len_total, device=query_states.device).unsqueeze(0)

    # 我们需要构建覆盖整个 Cache (key_states) 的 full_position_ids。
    # 策略：假设 Cache 是连续的滑动窗口，以当前片段的结束位置为基准，向前倒推。
    # 这样即使中间被 Prune 了，剩下的 Token 也会被强制“重映射”到相对于当前时刻连续的位置上。
    
    seq_len_total = key_states.shape[2]      # 总长度 (Past + Current)
    seq_len_curr = query_states.shape[2]     # 当前输入长度
    
    # 取当前片段的最后一个位置 ID
    current_end_pos = current_position_ids[0, -1].item()
    
    # 倒推开始位置： End - Total + 1
    start_pos = current_end_pos - seq_len_total + 1
    
    # 生成连续的 full_position_ids [batch, total_len]
    full_position_ids = torch.arange(
        start_pos, 
        current_end_pos + 1, 
        device=query_states.device
    ).unsqueeze(0)
    
    # 确保维度匹配 (Batch Size)
    if full_position_ids.shape[0] != query_states.shape[0]:
        full_position_ids = full_position_ids.repeat(query_states.shape[0], 1)

    # 现场计算 cos, sin (使用注入的 rotary_emb)
    # self.rotary_emb 是我们在 apply_debug_patch 时注入进来的
    cos, sin = self.rotary_emb(value_states, full_position_ids)
    
    # 应用 RoPE
    # query_states: 需要使用对应当前部分的 cos/sin
    # key_states:   需要使用全量的 cos/sin
    
    # 由于 cos/sin 是全量的 [batch, total_len, head_dim]，我们需要切片给 query 用
    # query 对应的是最后 seq_len_curr 个位置
    cos_q = cos[:, -seq_len_curr:, :]
    sin_q = sin[:, -seq_len_curr:, :]
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q, cos, sin)

    # =========================================================================
    # 后续逻辑保持标准 Attention (GQA -> Score -> Softmax -> Output)
    # =========================================================================

    # GQA 处理
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 计算 Attention Scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        # Handle 4D masks where K dimension may differ from Q
        if attention_mask.shape[-1] != attn_weights.shape[-1]:
            attention_mask = attention_mask[:, :, :, :attn_weights.shape[-1]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    # Return only 2 values for transformers >= 5.x compatibility
    # (past_key_values is updated in-place via past_key_values.update() above)
    return attn_output, attn_weights

# -------------------------------------------------------------------------
# 辅助函数：修改 apply_rotary_pos_emb 以支持不同的 cos/sin 长度
# -------------------------------------------------------------------------
def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k, unsqueeze_dim=1):
    """
    自定义的 RoPE 应用函数，允许 Q 和 K 使用不同的 cos/sin 切片。
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 调整维度以支持广播
    cos_q = cos_q.unsqueeze(unsqueeze_dim)
    sin_q = sin_q.unsqueeze(unsqueeze_dim)
    cos_k = cos_k.unsqueeze(unsqueeze_dim)
    sin_k = sin_k.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

# -------------------------------------------------------------------------
# Patch 应用函数
# -------------------------------------------------------------------------
def apply_debug_patch(model):
    print("Applying Raw-Cache RoPE Patch to Qwen2Attention...")
    
    # 1. 找到 Model 的 Rotary Embedding 模块
    # Qwen2 通常结构: model.model.rotary_emb 或者 model.rotary_emb
    # 我们遍历查找最稳妥
    rotary_emb_module = None
    for name, module in model.named_modules():
        if "rotary_emb" in name and isinstance(module, nn.Module):
            rotary_emb_module = module
            break
            
    if rotary_emb_module is None:
        # 尝试硬编码路径
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            rotary_emb_module = model.model.rotary_emb
        elif hasattr(model, "rotary_emb"):
            rotary_emb_module = model.rotary_emb
            
    if rotary_emb_module is None:
        raise ValueError("Could not find 'rotary_emb' module in the model! Patch failed.")

    print(f"Found Rotary Embedding Module: {rotary_emb_module}")

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2Attention):
            # 2. 将 rotary_emb 注入到 Attention 层中，方便 forward 调用
            module.rotary_emb = rotary_emb_module
            
            # 3. 替换 forward 方法
            module.forward = types.MethodType(qwen2_raw_cache_attention_forward, module)
            count += 1
            
    print(f"Patched {count} layers. \nMode: [Raw KV Cache] -> [On-the-fly RoPE calculation].")