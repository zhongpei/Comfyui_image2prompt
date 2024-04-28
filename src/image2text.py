from .moondream_model import MoondreamModel
from .moondream2_model import Moodream2Model
from .internlm_model import InternlmVLModle
from .uform_qwen_model import UformQwenModel
from .deepseek_model import DeepseekVLModel
from .wd_v3_model import WdV3Model
from .llama3_model import Llama3vModel
from PIL import Image
import numpy as np
from .utils import remove_specific_patterns,tensor2pil
from tqdm import tqdm
GLOBAL_WdV3Model = None

class LoadImage2TextModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "moondream1",
                    "moondream2", 
                    "bunny-llama3-8b-v",
                    "internlm-xcomposer2-vl-7b", 
                    "uform-qwen",
                    "wd-swinv2-tagger-v3",
                    "deepseek-vl-1.3b-chat",
                    "deepseek-vl-7b-chat",                    
                    ], {"default": "moondream2"}),
                "device": (["cpu", "cuda", ], {"default": "cuda"}),
                "low_memory": ("BOOLEAN", {"default": False}),
            }
        }


    RETURN_TYPES = ("IMAGE2TEXT_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo🐼/image2prompt"

    def get_model(self, model, device, low_memory):       
            
        if model == 'internlm-xcomposer2-vl-7b':
            return (InternlmVLModle(device=device,low_memory=low_memory),)
        elif model == 'uform-qwen':
            return (UformQwenModel(device=device,low_memory=low_memory),)
        elif model == "moondream2":
            return (Moodream2Model(device=device,low_memory=low_memory),)
        elif model == "wd-swinv2-tagger-v3":
            return (WdV3Model(device=device,low_memory=low_memory),)
        elif model == "deepseek-vl-1.3b-chat" or model == "deepseek-vl-7b-chat":
            return (DeepseekVLModel(device=device,low_memory=low_memory,model_name=model), )
        elif model == "bunny-llama3-8b-v":
            return (Llama3vModel(device=device,low_memory=low_memory),)
        
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
                "print_log": ("BOOLEAN", {
                    "default": False,
                    
                }),
            }
        }

    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "fofo🐼/image2prompt"

    def get_value(self, model, image, query, custom_query, print_log):
        # Ensure custom queries are prioritized
        if len(custom_query) > 0:
            query = custom_query
        # Initialize the list of strings to return
        answers = []
        # Iterate over each batch of images
        with tqdm(total=len(image)) as pbar:
            for img in image:
                # Convert Tensor to image
                
                img = tensor2pil(img)
                img = img.convert("RGB")
                # Additional processing for specific models
                if model.name == "internlm":
                    query = f"<ImageHere>{query}"
                
                result = model.answer_question(img, query)
                if print_log:
                    print(result)
                # Call the answer_question method for each batch of images and add to the answers list
                answers.append(result)
                pbar.update(1)

        # Return a list of strings, each corresponding to an answer for a batch
        return (answers,)

class Image2TextWithTags:
    @classmethod
    def INPUT_TYPES(cls):
        result = Image2Text.INPUT_TYPES()
        result["required"].update(
            {
                "score": ("BOOLEAN", {
                    "default": False,
                    
                }),
                "remove_1girl": ("BOOLEAN", {
                    "default": True,
                    
                }),

            },
        ) 
        return result
    
    OUTPUT_IS_LIST = (True,True,True)
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ('FULL PROMPT', "PROMPT", "TAGS")
    FUNCTION = "get_value"
    CATEGORY = "fofo🐼/image2prompt"

    def get_value(self, model, image, query, custom_query, print_log, score,remove_1girl):
        global GLOBAL_WdV3Model
        if GLOBAL_WdV3Model is None:
            GLOBAL_WdV3Model=WdV3Model(device="cpu",low_memory=False)

        
        result2 =  Image2Text().get_value(GLOBAL_WdV3Model, image, query, "score" if score is True else "",print_log)
        
            
        result1 =  Image2Text().get_value(model, image, query, custom_query,print_log)
        output = []
        for r1 in result1[0]:
            for r2 in result2[0]:
                if remove_1girl:
                    r2 = remove_specific_patterns(r2)
                output.append(f"{r1}\n\n{r2}")

        return (output, result1[0], result2[0])
    




