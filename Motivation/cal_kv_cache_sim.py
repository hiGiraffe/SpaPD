import torch
from torch.nn.functional import cosine_similarity
import os


device = 'cpu'
# 加载.pth文件
result_dir = 'kv_cache'
os.makedirs(result_dir, exist_ok=True)

ori_cache = torch.load(os.path.join(result_dir, "llama_query_full_kv_cache.pth"), weights_only=False)
sparse_cache = torch.load(os.path.join(result_dir, "llama_query_sparge_kv_cache.pth"), weights_only=False)

# 提取张量
'''
It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
'''

# 提取张量
k1_list, v1_list = ori_cache.key_cache, ori_cache.value_cache  # List[Tensor]
k2_list, v2_list = sparse_cache.key_cache, sparse_cache.value_cache  # List[Tensor]

# 确保层数一致
assert len(k1_list) == len(k2_list), "层数不一致"
assert len(v1_list) == len(v2_list), "层数不一致"

# 计算每层的余弦相似度
cos_sim_k_list, cos_sim_v_list = [], []

for layer_idx in range(len(k1_list)):
    k1, k2 = k1_list[layer_idx], k2_list[layer_idx]
    v1, v2 = v1_list[layer_idx], v2_list[layer_idx]

    # print(f"第 {layer_idx} 层的 K 张量形状不一致, 原始K的维度: {k1.shape}, 稀疏化K的维度: {k2.shape}")

    # 确保形状一致
    assert k1.shape == k2.shape, f"第 {layer_idx} 层的 K 张量形状不一致, 原始K的维度: {k1.shape}, 稀疏化K的维度: {k2.shape}"
    assert v1.shape == v2.shape, f"第 {layer_idx} 层的 V 张量形状不一致, 原始V的维度: {v1.shape}, 稀疏化K的维度: {v2.shape}"

    # 展平张量
    k1_flat = k1.reshape(-1, k1.size(-1)).to(device)  # 使用 reshape 替代 view
    k2_flat = k2.reshape(-1, k2.size(-1)).to(device)
    v1_flat = v1.reshape(-1, v1.size(-1)).to(device)
    v2_flat = v2.reshape(-1, v2.size(-1)).to(device)

    # 计算余弦相似度
    cos_sim_k = cosine_similarity(k1_flat, k2_flat, dim=1).mean()  # 平均值
    cos_sim_v = cosine_similarity(v1_flat, v2_flat, dim=1).mean()

    # 存储每层的结果
    cos_sim_k_list.append(cos_sim_k.item())
    cos_sim_v_list.append(cos_sim_v.item())

# 计算所有层的平均余弦相似度
avg_cos_sim_k = sum(cos_sim_k_list) / len(cos_sim_k_list)
avg_cos_sim_v = sum(cos_sim_v_list) / len(cos_sim_v_list)

print(f"K 的平均余弦相似度: {avg_cos_sim_k:.4f}")
print(f"V 的平均余弦相似度: {avg_cos_sim_v:.4f}")