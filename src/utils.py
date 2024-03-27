
from PIL import Image
import numpy as np
import torch
import re
import os





def pil2tensor(image):
  return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))






def remove_specific_patterns(text):
    # 构建一个匹配特定词汇前面有数字的正则表达式
    pattern = r"\d+(girl|boy|man|person|woman|men|women|girls|boys|mans),*\b"
    # 使用正则表达式去除匹配的模式
    modified_text = re.sub(pattern, "", text)

    return modified_text

def is_bf16_supported():
    # 首先检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. BF16 is not supported.")
        return False

    # 获取默认CUDA设备的ID
    device_id = torch.cuda.current_device()
    # 获取设备属性
    device_properties = torch.cuda.get_device_properties(device_id)
    
    # 在较新的PyTorch版本中，可以直接通过属性检查支持
    # 注意：这个属性可能在不同版本的PyTorch中有所不同
    # 请根据您的PyTorch版本进行调整

    bf16_supported_gpus = [
        'NVIDIA A100',
        'NVIDIA V100',
        'NVIDIA GeForce RTX 3060',  
        'NVIDIA GeForce RTX 3070',  
        'NVIDIA GeForce RTX 3080',  
        'NVIDIA GeForce RTX 3090',  
        'NVIDIA GeForce RTX 4090',  
        "NVIDIA GeForce RTX 4060 Ti",
        "NVIDIA GeForce RTX 4060",
        "NVIDIA GeForce RTX 4070",
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4080 Ti"
        # 根据需要添加更多支持bf16的显卡型号
    ]
    if hasattr(device_properties, 'supports_bfloat16'):
        return device_properties.supports_bfloat16
    
    # 对于旧版本的PyTorch，可能需要根据具体的GPU型号来手动判断
    # 或者假定最新的几代GPU支持BF16
    gpu_model = torch.cuda.get_device_name(0)
    if any(supported_gpu in gpu_model for supported_gpu in bf16_supported_gpus):
        print(f"{gpu_model} supports bf16.")
        return True
    
    print(f"{gpu_model} Unable to directly check BF16 support. Please check your PyTorch version and GPU specs.")
    return False


def download_model(repo:str)->str:
    from .install import get_model_dir
    from huggingface_hub import snapshot_download
    local_dir = get_model_dir("{}".format(repo.replace("/","__")))
    
    if os.path.exists(local_dir):
        model_path = local_dir
    else:
        model_path = snapshot_download(repo, local_dir=local_dir)
    return model_path