from transformers import AutoModel, AutoProcessor
import torch
import os
from huggingface_hub import snapshot_download


from .install import get_model_dir
from .utils import pil2tensor





class UformQwenModel():
    def __init__(self, device="cpu", low_memory=False):

        repo="unum-cloud/uform-gen2-qwen-500m"
        self.name = "uform-qwen"
        local_dir = get_model_dir("uform-gen2-qwen-500m")
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir)

        



        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if torch.cuda.is_available() and device == "cuda":
            self.model = AutoModel.from_pretrained(                
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,                
                device_map="auto",
                max_length=1024,
            ).to(device).eval()

            
        else:
            self.model = AutoModel.from_pretrained(                
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,
                max_length=1024,
            ).to(device).float().eval()
            
        

    def answer_question(self, image, question):     
        

        inputs = self.processor(text=[question], images=[image], return_tensors="pt")
        if self.device == "cuda":
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=256,
                eos_token_id=151645,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0]
        decoded_text = decoded_text.replace("<|im_end|>","")
        return decoded_text

