# https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI
import torch
import comfy.model_management as model_management
from copy import deepcopy
from .bnk_adv_encode import advanced_encode
def maximum_absolute_values(tensors,reversed=False):
    shape = tensors.shape
    tensors = tensors.reshape(shape[0], -1)
    tensors_abs = torch.abs(tensors)
    if not reversed:
        max_abs_idx = torch.argmax(tensors_abs, dim=0)
    else:
        max_abs_idx = torch.argmin(tensors_abs, dim=0)
    result = tensors[max_abs_idx, torch.arange(tensors.shape[1])]
    return result.reshape(shape[1:])

def get_closest_token_cosine_similarities(single_weight, all_weights, return_scores=False):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(all_weights.device))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_id_list = sorted_ids.tolist()
    if not return_scores:
        return best_id_list
    scores_list = sorted_scores.tolist()
    return best_id_list, scores_list

def get_single_cosine_score(single_weight,concurrent_weight):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    score = cos(concurrent_weight.unsqueeze(0), single_weight.unsqueeze(0)).item()
    return score

def refine_token_weight(token_id, all_weights, sculptor_method, sculptor_multiplier):
    initial_weight = all_weights[token_id]
    pre_mag = torch.norm(initial_weight)
    concurrent_weights_ids, scores = get_closest_token_cosine_similarities(initial_weight,all_weights,True)
    concurrent_weights_ids, scores = concurrent_weights_ids[1:], scores[1:]
    
    previous_cos_score = 0
    cos_score = 1
    iter_num = 0
    s = []
    tmp_weights = []
    ini_w = torch.clone(initial_weight)

    while previous_cos_score < cos_score:
        if iter_num > 0:
            previous_cos_score = cos_score
        s.append(scores[iter_num])
        current_weight = all_weights[concurrent_weights_ids[iter_num]]
        tmp_weights.append(current_weight)
        vec_sum = torch.sum(torch.stack(tmp_weights),dim=0)
        cos_score = get_single_cosine_score(ini_w, vec_sum)
        iter_num += 1
    del s[-1]
    del tmp_weights[-1]

    if len(s) <= 1: return initial_weight.cpu(), 0

    if sculptor_method == "maximum_absolute":
        concurrent_weights = torch.stack([ini_w/torch.norm(ini_w)]+[t/torch.norm(t) for i, t in enumerate(tmp_weights)])
        initial_weight = maximum_absolute_values(concurrent_weights)
        initial_weight *= pre_mag / torch.norm(initial_weight)
        return initial_weight.cpu(), len(s)

    concurrent_weights = torch.sum(torch.stack([t * s[i]**2 for i, t in enumerate(tmp_weights)]), dim=0)
    final_score = get_single_cosine_score(initial_weight,concurrent_weights) * sculptor_multiplier

    if sculptor_method == "backward":
        initial_weight = initial_weight + concurrent_weights * final_score
    elif sculptor_method == "forward":
        initial_weight = initial_weight - concurrent_weights * final_score
        
    initial_weight *= pre_mag / torch.norm(initial_weight)
    return initial_weight.cpu(), len(s)

def vector_sculptor_tokens(clip, text, sculptor_method, token_normalization, sculptor_multiplier):
    ignored_token_ids = [49406, 49407, 0]
    initial_tokens = clip.tokenize(text)
    total_found = 0
    total_replaced = 0
    total_candidates = 0
    
    for k in initial_tokens:
        mean_mag = 0
        mean_mag_count = 0
        to_mean_coords = []
        clip_model = getattr(clip.cond_stage_model, f"clip_{k}", None)
        all_weights = torch.clone(clip_model.transformer.text_model.embeddings.token_embedding.weight).to(device=model_management.get_torch_device())
        if token_normalization == "mean of all tokens":
            all_mags = torch.stack([torch.norm(t) for t in all_weights])
            mean_mag_all_weights = torch.mean(all_mags, dim=0).item()
        for x in range(len(initial_tokens[k])):
            for y in range(len(initial_tokens[k][x])):
                token_id, attn_weight = initial_tokens[k][x][y]
                if token_id not in ignored_token_ids and sculptor_multiplier > 0:
                    total_candidates += 1
                    new_vector, n_found = refine_token_weight(token_id,all_weights, sculptor_method, sculptor_multiplier)
                    if n_found > 0:
                        total_found += n_found
                        total_replaced += 1
                else:
                    new_vector = all_weights[token_id]
                # if y not in [0,76] and token_normalization != "none":
                if token_normalization != "none":
                    if token_normalization == "mean" or token_normalization == "mean * attention":
                        mean_mag += torch.norm(new_vector).item()
                        mean_mag_count += 1
                        to_mean_coords.append([x,y])
                    elif token_normalization == "set at 1":
                        new_vector /= torch.norm(new_vector)
                    elif token_normalization == "default * attention":
                        new_vector *= attn_weight
                    elif token_normalization == "set at attention":
                        new_vector /= torch.norm(new_vector) * attn_weight
                    elif token_normalization == "mean of all tokens":
                        new_vector /= torch.norm(new_vector) * mean_mag_all_weights
                initial_tokens[k][x][y] = (new_vector, attn_weight)
        if (token_normalization == "mean" or token_normalization == "mean * attention") and mean_mag_count > 0:
            mean_mag /= mean_mag_count
            for x, y in to_mean_coords:
                token_weight, attn_weight = initial_tokens[k][x][y]
                if token_normalization == "mean * attention":
                    twm = attn_weight
                else:
                    twm = 1
                token_weight = token_weight / torch.norm(token_weight) * mean_mag * twm
                initial_tokens[k][x][y] = (token_weight, attn_weight)
        del all_weights
    if total_candidates > 0:
        print(f"total_found: {total_found} / total_replaced: {total_replaced} / total_candidates: {total_candidates} / candidate proportion replaced: {round(100*total_replaced/total_candidates,2)}%")
    return initial_tokens



def add_to_first_if_shorter(conditioning1,conditioning2,x=0):
    min_dim = min(conditioning1[x][0].shape[1],conditioning2[x][0].shape[1])
    if conditioning2[x][0].shape[1]>conditioning1[x][0].shape[1]:
        conditioning2[x][0][:,:min_dim,...] = conditioning1[x][0][:,:min_dim,...]
        conditioning1 = conditioning2
    return conditioning1

# cheap slerp / I will bet an eternity doing regex that this is the dark souls 2 camera direction formula
def average_and_keep_mag(v1,v2,p1):
    m1 = torch.norm(v1)
    m2 = torch.norm(v2)
    v0 = v1 * p1 + v2 * (1 - p1)
    v0 = v0 / torch.norm(v0) * (m1 * p1 + m2 * (1 - p1))
    return v0

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(high, low, val):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)
    
class PromptConditioning:
    """
    sculptor_intensity:效果的强度。如果方向没有反转，最多到3，你将保留你提示的总体意义。如果反转，则不要超过1~2.5，否则随机性增强。
    sculptor_method:
        forward：减去最近的向量。超过1可能会有逆效果。
        backward：相反，加上它们。
        maximum_absolute：规范化向量并选择最远离0的值。除了设置为0时禁用外，强度在这里没有效果。这倾向于使简单主题的构成更复杂，复杂提示更混乱。根据情况可能有益也可能没有。它主要是为了好玩，但对于抽象概念可以产生极好的结果。
    token_normalization：重新工作每个向量的大小。
    建议要么“无”保留默认设置，要么“平均”设置每个令牌的重要性为它们的整体平均值。
    “设置为1”会将它们全部设置为1，我不知道这是否是个好主意。
    “所有令牌的平均值”将取预设条件权重内所有向量的平均值，这可能是个坏主意，但也为什么不呢。
    如果强度设置为0，令牌的规范化仍然有效。将其设置为0并选择“无”将返回默认的舒适条件。

    无论主题如何，两个方向都提供有效的变化。

    对于一般用途，我推荐向后设置0.5用于正面提示，并对于负面提示“保持原位”。

    将令牌大小规范化为它们的平均值似乎也有积极的效果。特别是对于负面提示，这似乎降低了毁坏图像的比例。
    merge_conditioning_type：球面线性插值，平均
    merge_conditioning_strength_custom： “分别设置\\n\\n换行的prompt的值，用逗号分割，数量为分段数量-1”
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {"multiline": True}),
                "merge_conditioning_type":(["slerp","average"], {"default": "average"}),
                "merge_conditioning_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "merge_conditioning_strength_custom": ("STRING", {"multiline": True} ),
                "sculptor_intensity": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.1}),
                "sculptor_method" : (["forward","backward","maximum_absolute"],{"default":"backward"}),
                "token_normalization": (["none", "mean", "set at 1", "default * attention", "mean * attention", "set at attention", "mean of all tokens"],),
 
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "fofo🐼/conditioning"

    def exec(
            self, 
            clip, 
            text,
            merge_conditioning_type, 
            merge_conditioning_strength,
            merge_conditioning_strength_custom, 
            sculptor_method, 
            token_normalization, 
            sculptor_intensity,
            ):
        cond_list = []
        prompt_list = []

        for line in text.split("\n\n"):
            line = line.strip()
            if len(line) == 0:
                continue
            prompt_list.append(line)

        if merge_conditioning_strength_custom is not None and merge_conditioning_strength_custom != "":
            merge_conditioning_strength = [float(x) for x in merge_conditioning_strength_custom.split(",")]
            if len(merge_conditioning_strength) != len(prompt_list) - 1:
                raise ValueError(f"merge_conditioning_strength_custom should have {len(prompt_list) - 1} values")
        else:
            merge_conditioning_strength=[merge_conditioning_strength] * (len(prompt_list) - 1)
        print(f"merge_conditioning_strength: {merge_conditioning_strength}")
        
        for line in prompt_list:

            sculptor_tokens = vector_sculptor_tokens(clip, line, sculptor_method, token_normalization, sculptor_intensity)
            cond, pooled = clip.encode_from_tokens(sculptor_tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]
            cond_list.append(conditioning)

        if len(cond_list) == 1:
            return (cond_list[0],)
        

        
        
        if merge_conditioning_type == "slerp":
            merge_func = slerp
        else:
            merge_func = average_and_keep_mag
            merge_conditioning_strength=[x*2 for x in merge_conditioning_strength]
        
        cond1 = deepcopy(cond_list[0])
        for i in range(1, len(cond_list)):
            cond2 = deepcopy(cond_list[i])  # 当前循环的 cond2 为 cond_list 的第i号元素
            # 进行原先的处理逻辑
            for x in range(min(len(cond1), len(cond2))):
                min_dim = min(cond1[x][0].shape[1], cond2[x][0].shape[1])
                if cond1[x][0].shape[2] == 2048:
                    cond1[x][0][:, :min_dim, :768] = merge_func(cond1[x][0][:, :min_dim, :768], cond2[x][0][:, :min_dim, :768], merge_conditioning_strength[i-1])
                    cond1[x][0][:, :min_dim, 768:] = merge_func(cond1[x][0][:, :min_dim, 768:], cond2[x][0][:, :min_dim, 768:], merge_conditioning_strength[i-1])
                else:
                    cond1[x][0][:, :min_dim, ...] = merge_func(cond1[x][0][:, :min_dim, ...], cond2[x][0][:, :min_dim, ...], merge_conditioning_strength[i-1])
                cond1 = add_to_first_if_shorter(cond1, cond2, x)
        return (cond1,)

class AdvancedCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP",),
            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
            # "affect_pooled": (["disable", "enable"],),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "fofo🐼/conditioning"

    def encode(self, clip, text, token_normalization, weight_interpretation, affect_pooled='disable'):
        embeddings_final, pooled = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0,
                                                   apply_to_pooled=affect_pooled == 'enable')
        return ([[embeddings_final, {"pooled_output": pooled}]],)
