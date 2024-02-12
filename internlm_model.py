import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


from .install import get_ext_dir
from .utils import pil2tensor

import time
from PIL import Image, ImageFile
import torch







class InternlmVLModle():
    def __init__(self, device="cpu", low_memory=False):
        if torch.cuda.is_available() and device == "cuda":
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        self.model_path, self.model, self.tokenizer = self.load_model(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.name = "internlm"
        self.low_memory = low_memory
        print(f"Model loaded on {device} with dtype {dtype} low_memory={low_memory}")


    @classmethod
    def load_model(cls, device, dtype):
        
        local_dir = get_ext_dir("model/internlm-xcomposer2-vl-7b")
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download("internlm/internlm-xcomposer2-vl-7b", local_dir=local_dir)
        #model_path = "internlm/internlm-xcomposer2-vl-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if torch.cuda.is_available() and device == "cuda":
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,
                device_map="auto"
            ).eval()

        else:
            model = model.cpu().float().eval()
        # `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，导致显存不足
        
        model.tokenizer = tokenizer
       
        
        return model_path,model,tokenizer
    
    def answer_question(self, image, question):     
        # example image
        import tempfile

        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir,"input.jpg")
        image.save(image_path)
        #image = pil2tensor(image)
        if self.device == "cuda":
            if self.low_memory:
                self.model.half().cuda().eval()
            with torch.cuda.amp.autocast(): 
                response, _ = self.model.chat(
                        query=question, 
                        image=image_path, 
                        tokenizer= self.tokenizer,
                        history=[], 
                        do_sample=True
                    )
            if self.low_memory:
                torch.cuda.empty_cache()
                print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                self.model.to("cpu", dtype=torch.float32)
                print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        else:
            response, _ = self.model.chat(
                    query=question,
                    image=image_path, 
                    tokenizer= self.tokenizer,
                    history=[], 
                    do_sample=True
                )
        print(response)
        return response