
from .src.install import check_and_install, check_and_install_version


check_and_install("tqdm")

check_and_install("image-reward",import_name="ImageReward")

check_and_install_version("Pillow","10.1.0",import_name="PIL")

check_and_install_version("huggingface_hub","0.20.1")

# 4bit internlm 
# linux must install manually for kernel compile
check_and_install_version("auto-gptq","0.7.1",import_name="auto_gptq")


check_and_install_version("einops","0.7.0")
check_and_install("torchvision")
check_and_install_version("accelerate","0.25.0")
check_and_install_version("timm","0.9.16")

# Qwen-1.5 awq
check_and_install_version("autoawq","0.2.3",import_name="awq")

## Qwen1_8 Prompt
check_and_install("tiktoken")
# check_and_install_version("transformers-stream-generator", "0.0.4",import_name="transformers_stream_generator")

# deepseek
check_and_install("attrdict")
check_and_install_version("einops","0.7.0")
check_and_install_version("sentencepiece","0.2.0")
check_and_install("git+https://github.com/deepseek-ai/DeepSeek-VL.git@86a3096",import_name="deepseek_vl")

# >= 4.37.1 Qwen-1.5      
# >= 4.38.2 deepseek , test ok == 4.37.1
# >= 4.38.2 llama3
check_and_install_version("transformers","4.38.2",up_version=False)

# llama3 4bit or 8bit
check_and_install_version("accelerate","0.29.3")
check_and_install_version("bitsandbytes","0.43.1")

# youdao Translate
check_and_install("pycryptodome",import_name="Crypto")

from .src.image2text import Image2Text, LoadImage2TextModel, Image2TextWithTags
from .src.text2prompt import LoadText2PromptModel,Text2Prompt,Text2GPTPrompt
from .src.tools import Translate2Chinese,ShowText,TextBox
from .src.conditioning import PromptConditioning,AdvancedCLIPTextEncode
from .src.reward import LoadImageRewardScoreModel,ImageRewardScore,ImageBatchToList
from .src.t5_gpt import T5_LLM_Node,QuantizationConfig_Node,Load_T5_LLM_Model

NODE_CLASS_MAPPINGS = {
    "Image2Text": Image2Text,
    "LoadImage2TextModel": LoadImage2TextModel,
    "Image2TextWithTags": Image2TextWithTags,
    "LoadText2PromptModel":LoadText2PromptModel,
    "Text2Prompt":Text2Prompt,
    "Text2GPTPrompt":Text2GPTPrompt,
    "Translate2Chinese|fofo":Translate2Chinese,
    "ShowText|fofo":ShowText,
    "TextBox|fofo":TextBox,
    "CLIP PromptConditioning|fofo":PromptConditioning,
    "CLIP AdvancedTextEncode|fofo":AdvancedCLIPTextEncode,
    "LoadImageRewardScoreModel|fofo":LoadImageRewardScoreModel,
    "ImageRewardScore|fofo":ImageRewardScore,
    "ImageBatchToList|fofo":ImageBatchToList,
    "LoadT5Model|fofo": Load_T5_LLM_Model,
    "T5QuantizationConfig|fofo": QuantizationConfig_Node,
    "T5Text2Prompt|fofo": T5_LLM_Node,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image2Text": "Image to Text 🐼",
    "LoadImage2TextModel": "Loader Image to Text Model 🐼",
    "Image2TextWithTags":"Image to Text with Tags 🐼",
    "LoadText2PromptModel":"Loader Text to Prompt Model 🐼",
    "Text2Prompt":"Text to Prompt 🐼",
    "Text2GPTPrompt":"Multi Text to GPTPrompt 🐼",
    "Translate2Chinese|fofo":"Translate Text to Chinese 🐼",
    "ShowText|fofo":"Show Text 🐼",
    "TextBox|fofo":"Text Box 🐼",
    "CLIP PromptConditioning|fofo":"CLIP Prompt Conditioning 🐼",
    "CLIP AdvancedTextEncode|fofo": "CLIP Advanced Text Encode 🐼",
    "LoadImageRewardScoreModel|fofo":"Load Image Reward Score Model 🐼",
    "ImageRewardScore|fofo":"Image Reward Score 🐼",
    "ImageBatchToList|fofo":"Image Batch to List 🐼",
    "LoadT5Model|fofo": "Load T5 Model 🐼",
    "T5QuantizationConfig|fofo": "T5 Quantization Config 🐼",
    "T5Text2Prompt|fofo": "T5 Text to Prompt 🐼",
}

## model dir
from .src.install import GLOBAL_MODELS_DIR
import os
if not os.path.exists(GLOBAL_MODELS_DIR):
    os.makedirs(GLOBAL_MODELS_DIR,exist_ok=True)



## js file
import folder_paths
import filecmp
import shutil
def copy_js_files():
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
    extentions_folder = os.path.join(folder_paths.base_path,"web" , "extensions" , "fofo" )
    os.makedirs(extentions_folder,exist_ok=True)
    
    result = filecmp.dircmp(javascript_folder, extentions_folder)
    print(javascript_folder)
    print(extentions_folder)
    if result.left_only or result.diff_files:
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extentions_folder, file)
            print(src_file,dst_file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)
copy_js_files()