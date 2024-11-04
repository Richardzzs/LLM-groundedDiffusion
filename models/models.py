import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler
from .unet_2d_condition import UNet2DConditionModel
from easydict import EasyDict
import numpy as np
# For compatibility
from utils.latents import get_unscaled_latents, get_scaled_latents, blend_latents
from utils import torch_device

# This is to be set in the `generate.py`
sd_key = ""
sd_version = ""
model_dict = None

def load_sd(key="stabilityai/stable-diffusion-2-1-base", use_fp16=False, load_inverse_scheduler=False, use_dpm_multistep_scheduler=False, scheduler_cls=None):
    """
    Keys:
     key = "CompVis/stable-diffusion-v1-4"
     key = "runwayml/stable-diffusion-v1-5"
     key = "stabilityai/stable-diffusion-2-1-base"
     
    Unpack with:
    ```
    model_dict = load_sd(key=key, use_fp16=use_fp16, **models.model_kwargs)
    vae, tokenizer, text_encoder, unet, scheduler, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict.scheduler, model_dict.dtype
    ```
    
    use_fp16: fp16 might have degraded performance
    use_dpm_multistep_scheduler: DPMSolverMultistepScheduler
    """
    
    # run final results in fp32
    if use_fp16:
        dtype = torch.float16
        revision = "fp16"
    else:
        dtype = torch.float
        revision = "main"
        
    vae = AutoencoderKL.from_pretrained(key, subfolder="vae", revision=revision, torch_dtype=dtype).to(torch_device)
    tokenizer = CLIPTokenizer.from_pretrained(key, subfolder="tokenizer", revision=revision, torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(key, subfolder="text_encoder", revision=revision, torch_dtype=dtype).to(torch_device)
    unet = UNet2DConditionModel.from_pretrained(key, subfolder="unet", revision=revision, torch_dtype=dtype).to(torch_device)
    if scheduler_cls is None: # Default setting (for compatibility)
        if use_dpm_multistep_scheduler:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(key, subfolder="scheduler", revision=revision, torch_dtype=dtype)
        else:
            scheduler = DDIMScheduler.from_pretrained(key, subfolder="scheduler", revision=revision, torch_dtype=dtype)
    else:
        print("Using scheduler:", scheduler_cls)
        assert not use_dpm_multistep_scheduler, "`use_dpm_multistep_scheduler` cannot be used with `scheduler_cls`"
        scheduler = scheduler_cls.from_pretrained(key, subfolder="scheduler", revision=revision, torch_dtype=dtype)
    
    if load_inverse_scheduler:
        inverse_scheduler = DDIMInverseScheduler.from_config(scheduler.config)
        model_dict.inverse_scheduler = inverse_scheduler

    model_dict = EasyDict(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=scheduler, dtype=dtype)

    return model_dict

def load_my_embeds(model_dict):
    tokenizer = model_dict.tokenizer
    text_encoder = model_dict.text_encoder
    # load the embeddings
    my_embeddings = torch.load("embeds/48/learned_embeds_final.bin")

    for placeholder_token, embed in my_embeddings.items():
        if embed.shape[0] != 1024:
           raise ValueError(f"Embed size for {placeholder_token} is {embed.shape[0]}, expected 1024.")

    # Add the placeholder token in tokenizer
    for placeholder_token, embed in my_embeddings.items():
        print(f"Adding token {placeholder_token}")
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

    text_encoder.resize_token_embeddings(len(tokenizer))
    for placeholder_token, embed in my_embeddings.items():
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = embed
    
    return model_dict

def encode_prompts(tokenizer, text_encoder, prompts, negative_prompt="", return_full_only=False, one_uncond_input_only=False):
    if negative_prompt == "":
        print("Note that negative_prompt is an empty string")
    
    text_input = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    
    max_length = text_input.input_ids.shape[-1]
    if one_uncond_input_only:
        num_uncond_input = 1
    else:
        num_uncond_input = len(prompts)
    uncond_input = tokenizer([negative_prompt] * num_uncond_input, padding="max_length", max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        cond_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    
    if one_uncond_input_only:
        return uncond_embeddings, cond_embeddings
    
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    
    if return_full_only:
        return text_embeddings
    return text_embeddings, uncond_embeddings, cond_embeddings

def process_input_embeddings(input_embeddings):
    assert isinstance(input_embeddings, (tuple, list))
    if len(input_embeddings) == 3:
        # input_embeddings: text_embeddings, uncond_embeddings, cond_embeddings
        # Assume `uncond_embeddings` is full (has batch size the same as cond_embeddings)
        _, uncond_embeddings, cond_embeddings = input_embeddings
        assert uncond_embeddings.shape[0] == cond_embeddings.shape[0], f"{uncond_embeddings.shape[0]} != {cond_embeddings.shape[0]}"
        return input_embeddings
    elif len(input_embeddings) == 2:
        # input_embeddings: uncond_embeddings, cond_embeddings
        # uncond_embeddings may have only one item
        uncond_embeddings, cond_embeddings = input_embeddings
        if uncond_embeddings.shape[0] == 1:
            uncond_embeddings = uncond_embeddings.expand(cond_embeddings.shape)
        # We follow the convention: negative (unconditional) prompt comes first
        text_embeddings = torch.cat((uncond_embeddings, cond_embeddings), dim=0)
        return text_embeddings, uncond_embeddings, cond_embeddings
    else:
        raise ValueError(f"input_embeddings length: {len(input_embeddings)}")
