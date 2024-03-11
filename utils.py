
from PIL import Image
import numpy as np
import torch
import re

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
    if hasattr(device_properties, 'supports_bfloat16'):
        return device_properties.supports_bfloat16
    else:
        # 对于旧版本的PyTorch，可能需要根据具体的GPU型号来手动判断
        # 或者假定最新的几代GPU支持BF16
        print("Unable to directly check BF16 support. Please check your PyTorch version and GPU specs.")
        return False


