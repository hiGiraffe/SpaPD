from modelscope import snapshot_download

# 指定模型名称
model_name = "ZhipuAI/CogVideoX-2b"

# 指定下载路径（绝对路径）
custom_path = "cogvideo"

# 下载模型
model_dir = snapshot_download(
    model_name,  # 模型名称
    subfolder="transformer",
    cache_dir=custom_path,  # 指定下载路径
)

print(f"模型已下载到：{model_dir}")