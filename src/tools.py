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
    CATEGORY = "fofo🐼/tools"

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

    CATEGORY = "fofo🐼/tools"

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
    CATEGORY = "fofo🐼/tools"

    def get_value(self, Text):
        return (Text,)