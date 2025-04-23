import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import triton
import triton.language as tl
from torch import Tensor


def hyperparameter_check(hyper, H, device):
    if type(hyper) == float or type(hyper) == int:
        hyper = torch.full((H,), float(hyper), device=device)
    elif isinstance(hyper, Tensor):
        assert len(hyper.shape) <= 1, "Hyperparameter tensor must be 1D"
        if len(hyper.shape) == 0:
            hyper = torch.full((H,), hyper.item(), device=device)
        assert hyper.numel() == H, f"Hyperparameter tensor must have {H} elements, but has {hyper.numel()}"
        hyper = hyper.to(device)
    else:
        # print(hyper)
        raise ValueError("Hyperparameter must be a float or a tensor")
    return hyper


@triton.jit
def triton_bmm_pool_sim_simmean(x_ptr, pool_ptr, sim_ptr, simthreshd1, N: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1 + h)
    x_fp32 = x.to(tl.float32)
    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)


def get_pool_sim_triton_simmean(x, block_size, simthreshd1):
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks


@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)
    

def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


@triton.jit
def triton_fill_causal_mask(mask, BqdivBk): #改过
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    # if k >= (q + 1) * BqdivBk: # 原版
    if k >= (K-Q) * BqdivBk + (q+1) * BqdivBk: # 修改版
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)

def fill_causal_mask_triton(mask, BqdivBk:float):
    assert mask.dim() == 2
    # print("BqdivBk:", BqdivBk)
    assert BqdivBk==1,"目前假定BlockQ=BlockK"
    print("mask shape:", mask.shape)
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)





def get_block_map_meansim(q, k, is_causal=False, BLKQ=128, BLKK=64, simthreshd1=0.3, cdfthreshd=0.999, is_sparse=True, return_lut=False, attention_sink=False):
    # print('q shape:', q.shape)       # [1, 32, 6, 128]       [batch, num_heads, seq_len, head_dim]
    # print('k shape:', k.shape)       # [1, 8, 6, 128]       [batch, key_head_nums, seq_len, head_dim]
    k = repeat_kv(k, q.shape[1] // k.shape[1])          # 实现GQA，将k重复组数
    # print('repeated k shape: ', k.shape)              # [batch, num_heads, seq_len, head_dim]
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    # print("pooled_qblocks:", pooled_qblocks)
    # print("sim_qblocks:", sim_qblocks)

    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)
    # print("sim_kblocks:", sim_kblocks)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    # print("sim_qblocks", sim_qblocks)
    # print(sim_qblocks.shape)
    
    # print('pooled_qblocks shape:', pooled_qblocks.shape)          # [batch, num_heads, block_q, head_dim]
    # print('pooled_kblocks shape:', pooled_kblocks.shape)          # [batch, num_heads, block_k, head_dim]
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
    pooled_score[~sim_kblocks] = -torch.inf

    # print("pooled score1:", pooled_score)
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        # print("causal mask:", causal_mask)
        pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)

    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    # print("pooled score2:", pooled_score)

    print("sorted score:", sorted_score.values)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
    cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~sim_kblocks] = 1
    final_map[~sim_qblocks] = 1

    # print("final map:", final_map)

    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    print("final map2:", final_map)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    return final_map
    

def apply_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    根据块掩码对Q、K、V矩阵进行掩码处理
    
    Args:
        q (Tensor): [batch, num_heads, seq_len, head_dim]
        k (Tensor): [batch, num_heads, seq_len, head_dim]
        v (Tensor): [batch, num_heads, seq_len, head_dim]
        mask (Tensor): [batch, num_heads, block_num, 1] 掩码张量（1保留，0舍弃）
    
    Returns:
        tuple: 处理后的Q、K、V张量
    """
    # 获取张量维度
    batch_size, num_heads, seq_len, head_dim = q.shape
    block_num = mask.shape[2]
    
    # 计算每个块的长度
    block_size = seq_len // block_num

    # 下面开始计算kv的mask
    num_key_values = k.shape[1]
    group = num_heads // num_key_values  # 计算扩展倍数
    
    # 步骤1：缩减mask的num_heads维度到num_key_values，对每个 num_key_values 的组内 n_rep 个值进行聚合（如取最大值或平均值），得到缩减后的mask [B, num_key_values, Bk, 1]。
    # 这里取最大值的意思就是一个group内有一个1那就是保留这个k
    k_mask = mask.view(batch_size, num_key_values, group, block_num, 1).max(dim=2)[0]

    # 将掩码扩展到序列长度维度
    q_mask = mask.repeat_interleave(block_size, dim=2)
    k_mask = k_mask.repeat_interleave(block_size, dim=2)

    # 确保扩展后的掩码形状正确
    assert q_mask.shape == (batch_size, num_heads, seq_len, 1), \
        f"扩展后的Q掩码形状不匹配: {q_mask.shape} vs 预期 {batch_size, num_heads, seq_len, 1}"
    assert k_mask.shape == (batch_size, num_key_values, seq_len, 1), \
        f"扩展后的Q掩码形状不匹配: {k_mask.shape} vs 预期 {batch_size,  num_key_values, seq_len, 1}"
    
    # 应用掩码（0位置会被置零）
    q = q * q_mask
    k = k * k_mask
    v = v * k_mask

    return q, k, v