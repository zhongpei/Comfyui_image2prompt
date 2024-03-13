import torch
from transformers import AutoModelForCausalLM
import os
from .install import get_ext_dir
from .utils import is_bf16_supported

import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

from huggingface_hub import snapshot_download

class DeepseekVLModel():
    def __init__(self, device="cpu", low_memory=False, model_name="deepseek-vl-7b-chat"):
        # specify the path to the model
        repo = f"deepseek-ai/{model_name}"
        self.name = f"{model_name}"
        local_dir = get_ext_dir(f"model/{model_name}")
        self.device = device
        if os.path.exists(local_dir):
            model_path = local_dir
        else:
            model_path = snapshot_download(repo, local_dir=local_dir,force_download=True, resume_download=False)
        
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.vl_chat_processor=vl_chat_processor
        self.tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if is_bf16_supported() and device == "cuda":
            self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        elif device == "cuda":
            self.vl_gpt = vl_gpt.to(torch.float16).cuda().eval()
        else:
            self.vl_gpt = vl_gpt.to(torch.float32).cpu().eval()
        self.device =device

    def answer_question(self, image, question):     
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}",
                "images": ["./images/training_pipelines.png"]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[image,],
            force_batchify=True
        )
        prepare_inputs=prepare_inputs.to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer
