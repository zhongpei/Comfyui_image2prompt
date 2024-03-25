

import folder_paths
import os,sys
import torch 
import comfy.utils

sys.path.append(os.path.join(__file__,'../../'))

from .utils import tensor2pil








class ImageBatchToList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_batch": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"

    CATEGORY = "fofo/Image"

    def run(self, image_batch):
        images = [image_batch[i:i + 1, ...] for i in range(image_batch.shape[0])]
        return (images, )
    
class LoadImageRewardScoreModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
                "device":(["cuda","cpu"],{"default":"cuda"})
            },
        }

    RETURN_TYPES = ("IMAGEREWARD_MODEL",)
    RETURN_NAMES = ("IMAGEREWARD_MODEL",)
    OUTPUT_NODE = True
    CATEGORY = "fofo"
    FUNCTION = "load_model"

    def load_model(self, device):       
        if device == "cuda" and not torch.cuda.is_available() :
            device = "cpu"       

        import ImageReward as RM
        model = RM.load("ImageReward-v1.0")
        model=model.to(device=device)
        
        return (model,)

class ImageRewardScore:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IMAGEREWARD_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True
                }),
                "top_k":("INT",{
                    "default": 3, 
                    "min": 1, #Minimum value
                    "max": 500, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","STRING","INT")
    RETURN_NAMES = ("IMAGES","SCORES_STR","SCORES_INT")
    OUTPUT_NODE = True
    CATEGORY = "fofo"
    FUNCTION = "reward"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,True,True)



    def reward(self, model, prompt, top_k , images):
        
        model = model[0]
        top_k = top_k[0]
        prompt = prompt[0]
        
        scores = []
        print(f"image count: {len(images)}")

        pbar = comfy.utils.ProgressBar(len(images))
        with torch.no_grad():

            for index in range(len(images)):
                score = model.score(prompt, tensor2pil(images[index]))
                scores.append(
                    {
                        "score":score,
                        "image":images[index]
                    }
                )
                print(f"{score:.2f}")
                pbar.update(1)

            
        scores = sorted(scores, key=lambda x: x['score'], reverse=True)

        images_r = []
        for x in scores:
            if top_k > 0:
                top_k= top_k - 1
                images_r.append(x['image'])
        
        scores_str=[str(x['score']) for x in scores]
        scores_int=[x['score'] for x in scores]


        return (images_r,scores_str,scores_int)

