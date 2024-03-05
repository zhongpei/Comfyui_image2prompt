from .moondream_model import MoondreamModel
from .moodream2_model import Moodream2Model
from .internlm_model import InternlmVLModle
from .uform_qwen_model import UformQwenModel
from PIL import Image
import numpy as np

class LoadImage2TextModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["moondream1","moondream2", "internlm-xcomposer2-vl-7b", "uform-qwen"], {"default": "moondream1"}),
                "device": (["cpu", "cuda", ], {"default": "cuda"}),
                "low_memory": ("BOOLEAN", {"default": False}),
            }
        }


    RETURN_TYPES = ("IMAGE2TEXT_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo"

    def get_model(self, model, device, low_memory):       
            
        if model == 'internlm-xcomposer2-vl-7b':
            return (InternlmVLModle(device=device,low_memory=low_memory),)
        elif model == 'uform-qwen':
            return (UformQwenModel(device=device,low_memory=low_memory),)
        elif model == "moondream2":
            return (Moodream2Model(device=device,low_memory=low_memory),)
        
        return (MoondreamModel(device=device,low_memory=low_memory),)
    
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

    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "fofo"

    def get_value(self, model, image, query, custom_query):
        # Ensure custom queries are prioritized
        if len(custom_query) > 0:
            query = custom_query
        # Initialize the list of strings to return
        answers = []
        # Iterate over each batch of images
        for img in image:
            # Convert Tensor to image
            img = Image.fromarray(
                np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
            )
            # Additional processing for specific models
            if model.name == "internlm":
                query = f"<ImageHere>{query}"

            result = model.answer_question(img, query)
            # Call the answer_question method for each batch of images and add to the answers list
            answers.append(result)

        # Return a list of strings, each corresponding to an answer for a batch
        return (answers,)
