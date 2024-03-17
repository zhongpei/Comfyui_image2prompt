from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import snapshot_download
from .install import get_model_dir
import torch


class QwenModel():
    def __init__(self, device="cpu", repo="Qwen/Qwen1.5-0.5B-Chat"):

        
        self.name = "qwen"
        local_dir = get_model_dir("{}".format(repo.replace("/","__")))
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.device = device

       

    def answer_question(self, question, system_prompt="You are a helpful assistant.", max_new_tokens=512,temperature=0.9):     

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
