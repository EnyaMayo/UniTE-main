from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 指定模型名称和下载路径
model_name = "Yukang/Llama-2-7b-longlora-32k-ft"
target_path = "/home/pshao8/poi/LLM4POI/weights/llama2"

# 下载模型到指定路径
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=target_path,      # 下载并存储到指定路径
    local_files_only=False,    # 允许在线下载
    device_map="cpu",          # 先加载到 CPU，避免 GPU 问题
    torch_dtype=torch.float32, # 使用 FP32，避免量化问题
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=target_path,
    local_files_only=False,
    use_fast=True,
)

print("Model downloaded and loaded successfully!")