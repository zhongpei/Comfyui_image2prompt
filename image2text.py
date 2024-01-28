from .model import Image2TextModel
from PIL import Image
import numpy as np

class LoadImage2TextModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            }
        }


    RETURN_TYPES = ("IMAGE2TEXT_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo"

    def get_model(self, device):
        return (Image2TextModel(device=device),)

class Image2Text:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGE2TEXT_MODEL", ),
                "image": ("IMAGE", "IMAGE_URL"),
                "query": ("STRING", {
                    "default": "What is this?",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "fofo"

    def get_value(self, model, image, query):
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
        return (model.answer_question(image,query),)