from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

import os


from huggingface_hub import snapshot_download
import re

from .install import get_model_dir
from .utils import pil2tensor, is_bf16_supported
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')




class Llama3vModel():
    def __init__(self, device="cpu", low_memory=False):

        repo="BAAI/Bunny-Llama-3-8B-V"
        self.name = "bunny-llama3-8b-v"
        local_dir = get_model_dir("bunny-llama3-8b-v")
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir)

        
        kwargs = {"device_map": "auto"}


        
        if low_memory:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        #else:
        #    kwargs['load_in_8bit'] = True
            
        if torch.cuda.is_available():
            #if is_bf16_supported:
            #    device, dtype = "cuda", torch.bfloat16
            #else:
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        
        self.device = device
        if device != "cuda":
            kwargs['device_map'] = {"": device}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True )
        if torch.cuda.is_available() and device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(                
                model_path, 
                torch_dtype=dtype, 
                trust_remote_code=True,              
                max_length=1024,
                **kwargs
            ).eval()

            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(                
                model_path, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                max_length=1024,

            ).float().eval()
        
        print(f"{repo} loaded on {device}. dtype {dtype}")
        

    def answer_question(self, image, question):   
        
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device=self.device)

        # image, sample images can be found in images folder

        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)

        # generate
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=512,
            use_cache=True
        )[0]

        return self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
