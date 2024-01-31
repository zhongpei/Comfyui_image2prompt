import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


from .install import get_ext_dir


import time
from PIL import Image, ImageFile
import torch







class InternlmVLModle():
    def __init__(self, device="cpu"):
        if torch.cuda.is_available() and device == "cuda":
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        self.model_path, self.model, self.tokenizer = self.load_model(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype



    @classmethod
    def load_model(cls, device, dtype):
        
        local_dir = get_ext_dir("model/internlm-xcomposer2-vl-7b")
        model_path = snapshot_download("internlm/internlm-xcomposer2-vl-7b", local_dir=local_dir)
        #model_path = "internlm/internlm-xcomposer2-vl-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if torch.cuda.is_available() and device == "cuda":
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                device_map="cuda"
            ).half().cuda().eval()

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
        if self.device == "cuda":
            with torch.cuda.amp.autocast(): 
                response, _ = self.model.chat(query=question, image=image_path, tokenizer= self.tokenizer,history=[], do_sample=False)
        else:
            response, _ = self.model.chat(query=question, image=image_path, tokenizer= self.tokenizer,history=[], do_sample=False)
        print(response)
        return response