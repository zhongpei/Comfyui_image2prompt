from .moondream_model import MoondreamModel
from .internlm_model import InternlmVLModle
from PIL import Image
import numpy as np

class LoadImage2TextModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["moondream1", "internlm-xcomposer2-vl-7b", ], {"default": "vikhyatk/moondream1"}),
                "device": (["cpu", "cuda", ], {"default": "cuda"}),
            }
        }


    RETURN_TYPES = ("IMAGE2TEXT_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo"

    def get_model(self, model, device):       
            
        if model == 'internlm-xcomposer2-vl-7b':
            return (InternlmVLModle(device=device),)
        
        return (MoondreamModel(device=device),)
class Image2Text:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGE2TEXT_MODEL", ),
                "image": ("IMAGE", "IMAGE_URL"),
                "query": (["Describe this photograph","What is this?","<ImageHere>Please describe this image in detail."], {
                    "default": "What is this?",
                    "multiline": False,
                }),
                "custom_query": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "fofo"

    def get_value(self, model, image, query, custom_query):
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
        if len(custom_query) > 0:
            query = custom_query
        return (model.answer_question(image,query),)