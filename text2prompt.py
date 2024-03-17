from .qwen_model import QwenModel


class LoadText2PromptModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([            
                    "hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt",        
                    "Qwen/Qwen1.5-0.5B-Chat",
                    "Qwen/Qwen1.5-1.8B-Chat", 
                    "Qwen/Qwen1.5-4B-Chat",
                    "Qwen/Qwen1.5-7B-Chat" 
                ], {"default": "hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt"}),
                "device": (["cpu", "cuda", ], {"default": "cpu"}),
                "low_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TEXT2PROMPT_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo🐼"

    def get_model(self, model, device, low_memory):     
            
        if low_memory and device=="cuda" and model.startswith("Qwen/"):
            model = f"{model}-AWQ"
        return (QwenModel(device=device, repo=model),)
    

_choice = ["YES", "NO"]
_system_prompts = [
    "You are a helpful assistant.",
    "你擅长翻译中文到英语。",
    "你擅长文言文翻译为英语。",
    "你是绘画大师，擅长描绘画面细节。",
    "你是剧作家，擅长创作连续的漫画脚本。"
]   


class Text2Prompt:

    """
    A custom node for text generation using GPT

    Attributes
    ----------
    max_tokens (`int`): Maximum number of tokens in the generated text.
    temperature (`float`): Temperature parameter for controlling randomness (0.2 to 1.0).
    top_p (`float`): Top-p probability for nucleus sampling.
    logprobs (`int`|`None`): Number of log probabilities to output alongside the generated text.
    echo (`bool`): Whether to print the input prompt alongside the generated text.
    stop (`str`|`List[str]`|`None`): Tokens at which to stop generation.
    frequency_penalty (`float`): Frequency penalty for word repetition.
    presence_penalty (`float`): Presence penalty for word diversity.
    repeat_penalty (`float`): Penalty for repeating a prompt's output.
    top_k (`int`): Top-k tokens to consider during generation.
    stream (`bool`): Whether to generate the text in a streaming fashion.
    tfs_z (`float`): Temperature scaling factor for top frequent samples.
    model (`str`): The GPT model to use for text generation.
    """
    def __init__(self):
        self.temp_prompt = ""
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"multiline": True} ),
                "model": ("TEXT2PROMPT_MODEL", {"default": ""}),
                "max_tokens": ("INT", {"default": 128}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0, "max": 1.0}),

                "print_output": (["enable", "disable"], {"default": "disable"}),
                "cached": (_choice,{"default": "NO"} ),
                "prefix": ("STRING", {"default": "must be in english and describe a picture according to follow the description below within 77 words: ", "multiline": True,}),
                "system_prompt": (_system_prompts, {"default": "You are a helpful assistant."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "fofo🐼"

    def generate_text(
            self,
            prompt, 
            max_tokens, 
            temperature, 

            model,
            print_output,
            cached,
            prefix,
            system_prompt
            ):

        if cached == "NO":           

            cont = model.answer_question( f"{prefix} {prompt}", system_prompt, max_new_tokens=max_tokens,temperature=temperature)
            self.temp_prompt  = cont.strip()
        else:
            cont = self.temp_prompt
        #remove fist 30 characters of cont
        try:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: {cont}")
            return {"ui": {"text": cont}, "result": (cont,)}

        except:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: ")
            return {"ui": {"text": " "}, "result": (" ",)}
        

class Text2GPTPrompt:
    def __init__(self):
        
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"multiline": True,"default":'You must use English and use the "Supplementary Description" content to add a more detailed picture description to the "Picture Description" within 77 words.'} ),
                "text1": ("STRING",{"multiline": True} ),
                "text2": ("STRING",{"multiline": True} ),
                "text1_perfix": ("STRING",{"multiline": True, "default":"Picture Description:"} ),
                "text2_perfix": ("STRING",{"multiline": True, "default":"Supplementary Description:" }),
                
                "print_output": (["enable", "disable"], {"default": "disable"}),

            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "fofo🐼"

    def generate_text(self, prompt,text1,text2,text1_perfix,text2_perfix,print_output):
        text1 = f"{text1_perfix}{text1}"
        text2 = f"{text1_perfix}{text2}"
        output = f"{prompt}\n\n{text1}\n\n{text2}\n\n"
        if print_output == "enable":
            print(output)
        return (output,)
    
class Translate2Chinese:
    def __init__(self):
        from .translate import Translater
        self.translate = Translater()
        
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",{"multiline": True} ),   
                "cache": (["enable", "disable"], {"default": "enable"}),             
                "print_output": (["enable", "disable"], {"default": "enable"}),

            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "fofo🐼"

    def generate_text(self, text, print_output,cache):

        if text is None  or text == "":
            return ("",)
        
        if text is not None:
            text = text.strip()
        
        if cache == "enable" and text in self.cache:
            return (self.cache[text],)
        try:
            output = self.translate.translate(text)
            self.cache[text]=output
        except Exception as err:
            print(err)
            output = ""
        if print_output == "enable":
            print(output)
        return (output,)
    
class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "fofo🐼"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

    
class TextBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "fofo🐼"

    def get_value(self, Text):
        return (Text,)