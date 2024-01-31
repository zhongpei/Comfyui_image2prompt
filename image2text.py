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
    QUERY_EXPERT_TAGS = """As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. 
"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IMAGE2TEXT_MODEL", ),
                "image": ("IMAGE", "IMAGE_URL"),
                "query": (["Describe this photograph.","What is this?","Please describe this image in detail.",cls.QUERY_EXPERT_TAGS], {
                    "default": "What is this?",
                    "multiline": True,
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

        if model.name == "internlm":
            query = f"<ImageHere>{query}"
        print(f"prompt: {query}")
        return (model.answer_question(image,query),)