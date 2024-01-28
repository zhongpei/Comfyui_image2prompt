



### 第1步：安装插件

最后，将本插件存储库克隆到您的ComfyUI custom_nodes目录中。使用以下命令克隆存储库：

```

git clone https://github.com/zhongpei/Comfyui-image2prompt

```

### 第2步：下载模型

首先，请从以下载 https://huggingface.co/vikhyatk/moondream1 到插件的model目录


可以使用镜像，请在终端中执行以下命令：


```

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download vikhyatk/moondream1 --local-dir custom_nodes/Comfyui-image2prompt/model

```