from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import torch

import os

import re
from huggingface_hub import snapshot_download


from .install import get_model_dir
from .utils import pil2tensor,is_bf16_supported






class WdV3Model():
    def __init__(self, device="cpu", low_memory=False):

        repo="p1atdev/wd-swinv2-tagger-v3-hf"
        self.name = "wd-swinv2-tagger-v3"
        local_dir = get_model_dir("wd-swinv2-tagger-v3-hf")
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir)




        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

        if torch.cuda.is_available() and device == "cuda":
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path,

                trust_remote_code=True,  
            ).to(device).eval()

            
        else:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path,

                trust_remote_code=True,  
            ).to(device).float().eval()
            
        

    def answer_question(self, image, question,):     

        def extract_threshold(text):
            # 使用正则表达式匹配模式
            match = re.search(r'threshold\s*=\s*([0-9]*\.?[0-9]+)', text)
            if match:
                return float(match.group(1))  # 返回匹配的数字部分
            else:
                return 0.35  # 如果没有找到匹配项，则返回None
            
        threshold=extract_threshold(question)
        #print(f"threshold:{threshold}")
        inputs = self.processor.preprocess(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs.to(self.model.device, self.model.dtype))
        logits = torch.sigmoid(outputs.logits[0]) # take the first logits

        # get probabilities
        results = {self.model.config.id2label[i]: logit.float() for i, logit in enumerate(logits)}
        results = {
            k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True) if v > threshold # 35% threshold
        }

        if "score" in question:
            return ", ".join([f"({k}:{v:.2f})" for k, v in results.items()]) if results else ""

        return ", ".join([f"{k}" for k, v in results.items()]) if results else ""

