from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os


from huggingface_hub import snapshot_download
import re

from .install import get_model_dir
from .utils import pil2tensor, is_bf16_supported





class Moodream2Model():
    def __init__(self, device="cpu", low_memory=False):

        repo="vikhyatk/moondream2"
        self.name = "moondream2"
        local_dir = get_model_dir("moondream2")
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir, revision="2024-03-05")

        

        if torch.cuda.is_available():
            if is_bf16_supported:
                device, dtype = "cuda", torch.bfloat16
            else:
                device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True ,revision="2024-03-05")
        if torch.cuda.is_available() and device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(                
                model_path, 
                torch_dtype=dtype, 
                trust_remote_code=True,              
                max_length=1024,
                revision="2024-03-05"
            ).to(device).eval()

            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(                
                model_path, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                max_length=1024,
                revision="2024-03-05"
            ).to(device).float().eval()
        
        print(f"{repo} loaded on {device}. dtype {dtype}")
        

    def answer_question(self, image, question):   
        enc_image = self.model.encode_image(image)
        result = self.model.answer_question(enc_image, question, self.tokenizer)
        clean_text = re.sub("<$|<END$", "", result)
        return clean_text

