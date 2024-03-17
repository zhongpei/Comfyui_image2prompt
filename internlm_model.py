import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


from .install import  get_model_dir
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
        if low_memory:
            self.model_path, self.model, self.tokenizer = self.load_4bit_model(device=device, dtype=dtype)
        else:
            self.model_path, self.model, self.tokenizer = self.load_model(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.name = "internlm"
        self.low_memory = low_memory
        
        print(f"Model loaded on {device} with dtype {dtype} low_memory={low_memory}")

    @classmethod
    def load_4bit_model(cls, device, dtype):
        import torch, auto_gptq
        from transformers import AutoModel, AutoTokenizer 
        from auto_gptq.modeling import BaseGPTQForCausalLM

        auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
        torch.set_grad_enabled(False)
        repo = 'internlm/internlm-xcomposer2-vl-7b-4bit'
        local_dir = get_model_dir("internlm-xcomposer2-vl-7b-4bit")
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir)
        class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
            layers_block_name = "model.layers"
            outside_layer_modules = [
                'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
            ]
            inside_layer_modules = [
                ["attention.wqkv.linear"],
                ["attention.wo.linear"],
                ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                ["feed_forward.w2.linear"],
            ]
        
        # init model and tokenizer
        model = InternLMXComposer2QForCausalLM.from_quantized(
            model_path, 
            trust_remote_code=True, 
            device_map="auto",
            max_length=1024,
            
        ).to(device=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        return model_path,model,tokenizer

    @classmethod
    def load_model(cls, device, dtype):
        
        local_dir = get_model_dir("internlm-xcomposer2-vl-7b")
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download("internlm/internlm-xcomposer2-vl-7b", local_dir=local_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if torch.cuda.is_available() and device == "cuda":    
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                trust_remote_code=True,                
                device_map="auto",
                max_length=1024,
            ).to(device).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,                
                max_length=1024,
            ).to(device).float().eval()
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

            with torch.cuda.amp.autocast(): 
                response, _ = self.model.chat(
                        query=question, 
                        image=image_path, 
                        tokenizer= self.tokenizer,
                        history=[], 
                        do_sample=True
                    )

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