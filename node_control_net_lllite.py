import math
import torch
import os

import comfy

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CATEGORY_NAME = "controlnet_lllite/"


def get_file_list(path):
    return [file for file in os.listdir(path) if file != "put_models_here.txt"]

def extra_options_to_module_prefix(extra_options):
    # extra_options = {'transformer_index': 2, 'block_index': 8, 'original_shape': [2, 4, 128, 128], 'block': ('input', 7), 'n_heads': 20, 'dim_head': 64}

    # block is: [('input', 4), ('input', 5), ('input', 7), ('input', 8), ('middle', 0),
    #   ('output', 0), ('output', 1), ('output', 2), ('output', 3), ('output', 4), ('output', 5)]
    # transformer_index is: [0, 1, 2, 3, 4, 5, 6, 7, 8], for each block
    # block_index is: 0-1 or 0-9, depends on the block
    # input 7 and 8, middle has 10 blocks

    # make module name from extra_options
    block = extra_options["block"]
    block_index = extra_options["block_index"]
    if block[0] == "input":
        module_pfx = f"lllite_unet_input_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    elif block[0] == "middle":
        module_pfx = f"lllite_unet_middle_block_1_transformer_blocks_{block_index}"
    elif block[0] == "output":
        module_pfx = f"lllite_unet_output_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    else:
        raise Exception("invalid block name")
    return module_pfx

class LLLitePatch:
    def __init__(self, modules, sigma_low, sigma_high, attn):
        self.modules = modules
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.attn = attn

    def __call__(self, q, k, v, extra_options):
        sigma = extra_options["sigmas"][0].item()
        cond_or_uncond = extra_options["cond_or_uncond"]

        if self.sigma_low <= sigma <= self.sigma_high:
            module_pfx = extra_options_to_module_prefix(extra_options) + self.attn
            if module_pfx + "_to_q" in self.modules:
                q = q + self.modules[module_pfx + "_to_q"](q, cond_or_uncond)
            if module_pfx + "_to_k" in self.modules:
                k = k + self.modules[module_pfx + "_to_k"](k, cond_or_uncond)
            if module_pfx + "_to_v" in self.modules:
                v = v + self.modules[module_pfx + "_to_v"](v, cond_or_uncond)

        return q, k, v
    
    def to(self, device):
        if hasattr(torch, "float8_e4m3fn") and device == torch.float8_e4m3fn:
            device = torch.float16 # or bfloat16?
        if hasattr(torch, "float8_e5m2") and device == torch.float8_e5m2:
            device = torch.float16
        
        for d in self.modules.keys():
            self.modules[d] = self.modules[d].to(device)
        return self

class LLLiteModule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        is_conv2d: bool,
        in_dim: int,
        depth: int,
        cond_emb_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d
        self.is_first = False

        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size*2
        if depth == 1:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            # kernel size 8は大きすぎるので、4にする / kernel size 8 is too large, so set it to 4
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = torch.nn.Sequential(*modules)

        if self.is_conv2d:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.down = torch.nn.Sequential(
                torch.nn.Linear(in_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim, in_dim),
            )

        self.depth = depth

    def set_cond(self, positive, negative):
        self.positive = positive
        self.negative = negative
        self.cond_emb = []

    def forward(self, x, cond_or_uncond=[0, 1]):

        if self.cond_emb == []:
            # print(f"cond_emb is None, {self.name}")
            self.cond_emb = []
            for cond_image in [self.positive["cond_image"], self.negative["cond_image"]]:
                cx = self.conditioning1(cond_image.to(x.device, dtype=x.dtype))
                if not self.is_conv2d:
                    # reshape / b,c,h,w -> b,h*w,c
                    n, c, h, w = cx.shape
                    cx = cx.view(n, c, h * w).permute(0, 2, 1)
                self.cond_emb.append(cx)

        multiplier = [self.positive["strength"], self.negative["strength"]]
        multipliers = torch.tensor([multiplier[i] for i in cond_or_uncond], device=x.device, dtype=x.dtype)

        if self.is_conv2d:
            multipliers = multipliers.view(-1, 1, 1, 1)
        else:
            multipliers = multipliers.view(-1, 1, 1)

        # multiplierが0の要素は計算省略したい気がするけど・・・めんどくさい
        cx = torch.cat([self.cond_emb[i] for i in cond_or_uncond])
        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        return cx * multipliers

class LLLiteModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_file_list(os.path.join(CURRENT_DIR, "models")),),
            }
        }

    RETURN_TYPES = ("LLLITE_MODEL",)
    FUNCTION = "load_lllite"
    CATEGORY = CATEGORY_NAME + "LLLiteModelLoader"

    def load_lllite(self, model_name):
        model_path = os.path.join(CURRENT_DIR, os.path.join(CURRENT_DIR, "models", model_name))
        ctrl_sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        
        # split each weights for each module
        module_weights = {}
        for key, value in ctrl_sd.items():
            fragments = key.split(".")
            module_name = fragments[0]
            weight_name = ".".join(fragments[1:])

            if module_name not in module_weights:
                module_weights[module_name] = {}
            module_weights[module_name][weight_name] = value

        # load each module
        modules_attn1 = {}
        modules_attn2 = {}
        for module_name, weights in module_weights.items():
            # ここの自動判定を何とかしたい
            if "conditioning1.4.weight" in weights:
                depth = 3
            elif weights["conditioning1.2.weight"].shape[-1] == 4:
                depth = 2
            else:
                depth = 1

            module = LLLiteModule(
                name=module_name,
                is_conv2d=weights["down.0.weight"].ndim == 4,
                in_dim=weights["down.0.weight"].shape[1],
                depth=depth,
                cond_emb_dim=weights["conditioning1.0.weight"].shape[0] * 2,
                mlp_dim=weights["down.0.weight"].shape[0],
            )
            info = module.load_state_dict(weights)
            if "attn1" in module_name:
                modules_attn1[module_name] = module
            else:
                modules_attn2[module_name] = module
        
        return ((modules_attn1, modules_attn2), )
    
class LLLiteConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LLLITE_CONDITIONING",)
    FUNCTION = "load_conditioning"
    CATEGORY = CATEGORY_NAME + "LLLiteConditioning"

    def load_conditioning(self, cond_image, strength):
        cond_image = cond_image.permute(0, 3, 1, 2)  # b,h,w,3 -> b,3,h,w
        cond_image = cond_image * 2.0 - 1.0  # 0-1 -> -1-+1
        return ({'cond_image': cond_image, 'strength': strength},)

class LLLiteApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lllite_model": ("LLLITE_MODEL",),
                "positive": ("LLLITE_CONDITIONING",),
                "negative": ("LLLITE_CONDITIONING",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY_NAME + "LLLiteApply"

    def apply(self, model, lllite_model, positive, negative, start_percent, end_percent):
        sigma_high = model.model.model_sampling.percent_to_sigma(start_percent / 100)
        sigma_low = model.model.model_sampling.percent_to_sigma(end_percent / 100)

        new_model = model.clone()
        for modules in lllite_model:
            for module in modules.values():
                module.set_cond(positive, negative)

        new_model.set_model_attn1_patch(LLLitePatch(lllite_model[0], sigma_low, sigma_high, "_attn1"))
        new_model.set_model_attn2_patch(LLLitePatch(lllite_model[1], sigma_low, sigma_high, "_attn2"))

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "LLLiteModelLoader": LLLiteModelLoader, 
    "LLLiteConditioning": LLLiteConditioning, 
    "LLLiteApply": LLLiteApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLLiteModelLoader": "LLLite Model Loader", 
    "LLLiteConditioning": "LLLite Conditioning", 
    "LLLiteApply": "LLLite Apply"
}
