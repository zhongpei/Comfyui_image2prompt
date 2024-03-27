from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BitsAndBytesConfig
)
import torch


from .utils import download_model,is_bf16_supported



class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class T5Model():
    def __init__(self,model,tokenizer,config,device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.config=config

class Load_T5_LLM_Model():
    @classmethod
    def INPUT_TYPES(cls):
        # Get a list of directories in the checkpoints_path


        return {
            "required": {
                "model": (["roborovski/superprompt-v1",], ),
                "device": (["cpu", "cuda", ], {"default": "cuda"}),

            },
            "optional": {                
                "quantizationConfig": ("QUANTIZATIONCONFIG",),
            }
        }    
    RETURN_TYPES = ("T5_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo🐼/prompt"
    def get_model(self,model,device,quantizationConfig=None):
        model_path = download_model(model)

        if torch.cuda.is_available():
            if is_bf16_supported:
                device, dtype = "cuda", torch.bfloat16
            else:
                device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        # Initialize model_kwargs without torch_dtype or other optional params
        model_kwargs = {
            'device_map': 'auto',
            'quantization_config': quantizationConfig,
            'trust_remote_code':True,
            'torch_dtype' : dtype,
        }

        # Load the model and tokenizer based on the model's configuration
        config = AutoConfig.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Dynamically loading the model based on its type
        if config.model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
        elif config.model_type in ["gpt2", "gpt_refact", "gemma"]:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        elif config.model_type == "bert":
            model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        model.to(device)
        return (T5Model(model=model,config=config,tokenizer=tokenizer,device=device),)


class T5_LLM_Node:
    @classmethod
    def INPUT_TYPES(cls):
        # Get a list of directories in the checkpoints_path


        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 777}),
                "model": ("T5_MODEL", ),
                "max_tokens": ("INT", {"default": 256, "min": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.1}),
            },

        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_NODE = False
    FUNCTION = "main"
    CATEGORY = "fofo🐼/prompt"

    def main(self, text, seed, model, max_tokens, temperature=1,top_p=0.9,top_k=50,repetition_penalty=1.2):
        torch.manual_seed(seed)

        generate_kwargs = {
            'max_length': max_tokens,
            'temperature':temperature, 
            'top_p':top_p, 
            'top_k':top_k, 
            'repetition_penalty':repetition_penalty
        }

        if model.config.model_type in ["t5", "gpt2", "gpt_refact", "gemma"]:
            input_ids = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            outputs = model.model.generate(input_ids, **generate_kwargs)
            generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return (generated_text,)
        elif model.config.model_type == "bert":
            return ("BERT model detected; specific task handling not implemented in this example.",)



class QuantizationConfig_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        quantization_modes = ["none", "load_in_8bit", "load_in_4bit"]
        return {
            "required": {
                "quantization_mode": (quantization_modes, {"default": "none"}),
                "llm_int8_threshold": ("FLOAT", {"default": 6.0}),
                "llm_int8_skip_modules": ("STRING", {"default": ""}),
                "llm_int8_enable_fp32_cpu_offload": ("BOOLEAN", {"default": False}),
                "llm_int8_has_fp16_weight": ("BOOLEAN", {"default": False}),
                "bnb_4bit_compute_dtype": ("STRING", {"default": "float32"}),
                "bnb_4bit_quant_type": ("STRING", {"default": "fp4"}),
                "bnb_4bit_use_double_quant": ("BOOLEAN", {"default": False}),
                "bnb_4bit_quant_storage": ("STRING", {"default": "uint8"}),
            }
        }
    
    FUNCTION = "main"
    CATEGORY = "fofo🐼/prompt"
    RETURN_TYPES = ("QUANTIZATIONCONFIG",)
    RETURN_NAMES = ("QuantizationConfig",)

    def main(self, quantization_mode, llm_int8_threshold: float = 6.0, llm_int8_skip_modules="", llm_int8_enable_fp32_cpu_offload=False, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype="float32", bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False, bnb_4bit_quant_storage="uint8"):

        llm_int8_skip_modules_list = llm_int8_skip_modules.split(',') if llm_int8_skip_modules else []

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantization_mode == "load_in_8bit",
            load_in_4bit=quantization_mode == "load_in_4bit",
            llm_int8_threshold=float(llm_int8_threshold),
            llm_int8_skip_modules=llm_int8_skip_modules_list,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype, torch.float32),
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,
        )

        return (quantization_config,)
