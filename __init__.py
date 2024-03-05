from .image2text import Image2Text, LoadImage2TextModel
from .install import check_and_install, check_and_install_version


check_and_install_version("Pillow","10.1.0")

check_and_install_version("huggingface_hub","0.20.1")
check_and_install_version("transformers","4.36.2")
check_and_install_version("einops","0.7.0")
check_and_install("torchvision")
check_and_install_version("accelerate","0.25.0")
check_and_install_version("timm","0.9.12")

NODE_CLASS_MAPPINGS = {
    "Image2Text": Image2Text,
    "LoadImage2TextModel": LoadImage2TextModel,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image2Text": "Image to Text",
    "LoadImage2TextModel": "Loader Image to Text Model",
}
