import argparse
import datetime
import inspect
import os
import random
import gc

from omegaconf import OmegaConf

import torch
torch.set_grad_enabled(False)

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# export PYTHONPATH="${PYTHONPATH}:/home/users/u6566739/project-004/XPaste"

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, save_videos_frames
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
import clip
import json


from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
from PIL import Image


VXPASTE_ACTION = {"thing": ["moving", "resting", "changing"], \
                "animal" : ["running", "walking", "sitting", "moving", "playing", "resting", "exploring", "sleeping", "eating", "jumping", "fighting"], \
                "bird" : ["flying", "chirping", "moving", "playing", "resting", "exploring", "sleeping", "eating", "flapping"], \
                "fish" : ["swimming","moving", "playing", "resting", "exploring", "sleeping", "eating" ]
                  }

# VXPASTE_ACTION_THINGS = ["moving"]
# VXPASTE_ACTION_ANIMALS = ["running", "walking", "sitting"]
# VXPASTE_ACTION_BIRDS = ["flying", "chirping"]
# VXPASTE_ACTION_FISH = ["swimming"]

VXPASTE_ADJECTIVES = {"fish": ["scaly", "iridescent", "long", "small", "metallic", "short", "big", "slimy", "smooth", "fat"], \
                    "animal" : ["big", "small", "old", "adult", "baby", "thick", "short", "tall", "long", "cute", "adorable", "furry",
                              "sleek", "weird"], \
                    "bird" : ["big", "small", "baby", "feathery", "fluffy", "cute", "shining", "plain", "croaking", "plumed",
                            "flightless", "clawed", "smooth", "adult", "old"], \
                    "thing" : ["big", "tall", "long", "short", "shiny", "clean", "dirty", "delicate", "sturdy",
                             "fragile", "luxurious", "sophisticated", "ultramodern", "traditional", "smooth", "clear", "simple", \
                               "beautiful", "ugly", "broken", "new", "old"]
                      }

# VXPASTE_ADJECTIVES_FISH = ["scaly", "iridescent", "long", "small", "metallic", "short", "big", "slimy", "smooth"]
#
# VXPASTE_ADJECTIVES_ANIMALS = ["big", "small", "old", "adult", "baby", "thick", "short", "tall", "long", "cute", "adorable", "furry",
#                               "sleek"]
# VXPASTE_ADJECTIVES_BIRDS = ["big", "small", "baby", "feathery", "fluffy", "cute", "shining", "plain", "croaking", "plumed",
#                             "flightless", "clawed"]
# VXPASTE_ADJECTIVES_THINGS = ["big", "small", "tall", "long", "short", "shiny", "clean", "dirty", "delicate", "sturdy",
#                              "fragile", "luxurious", "sophisticated", "ultramodern", "traditional", "smooth", "clear"]

# VXPASTE_ADJECTVES = { "airplane": ["long", "short", "big", "small", "jet", "propeller", "boeing", "airbus"],
#     "bear": ["black", "white", "big", "baby", "small", "thick", "angry"],
#     "bird": ["big", "small"],
#     "boat"
#
# }

YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane", "type": "thing"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear", "type": "animal"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird", "type": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat", "type": "thing"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car", "type": "thing"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat", "type": "animal"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow", "type": "animal"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer", "type": "animal"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog", "type": "animal"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck", "type": "bird"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal", "type": "animal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant", "type": "animal"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish", "type": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc", "type": "thing"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox", "type": "animal"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog", "type": "animal"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda", "type": "animal"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe", "type": "animal"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse", "type": "animal"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard", "type": "animal"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard", "type": "animal"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey", "type": "animal"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike", "type": "thing"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse", "type": "animal"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot", "type": "bird"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person", "type": "animal"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit", "type": "animal"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark", "type": "fish"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard", "type": "thing"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake", "type": "animal"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard", "type": "thing"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel", "type": "animal"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard", "type": "thing"},
    {"color": [255, 208, 186], "isthing": 1,
     "id": 34, "name": "tennis_racket", "type": "thing"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger", "type": "animal"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train", "type": "thing"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck", "type": "thing"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle", "type": "animal"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale", "type": "fish"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra", "type": "animal"},
]

# usage:
#python scripts/animate_bushfire.py - -config configs/prompts/5-RealisticVision.yaml


def show_images(batch: torch.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    pil_image=Image.fromarray(reshaped.numpy())
    #display(pil_image)
    return pil_image

def get_clip_scores(images: torch.Tensor, prompt, clip_model, preprocess):
    images = images.permute(0,2,1,3,4).squeeze(0)
    # print(f'images: {images.shape} {prompt}')

    # for batch_idx in range(images.shape[0]):
    text_feature = clip.tokenize(prompt).cuda()
    _, logits_per_text = clip_model(
        torch.stack([preprocess(show_images(i.unsqueeze(0))) for i in images], 0).cuda(), text_feature)
    logits_per_text = logits_per_text.view(-1).cpu().tolist()
    return logits_per_text

def make_prompts(object_metadata, num_prompts):
    obj_type = object_metadata["type"]
    obj_name = object_metadata["name"]
    # num_prompts = 10
    prompts = []
    for idx in range(num_prompts):
        adjective = random.choice(VXPASTE_ADJECTIVES[obj_type])
        action = random.choice(VXPASTE_ACTION[obj_type])
        prompts.append(f"a close up video of a {action} {adjective} {obj_name}, centred")
    return prompts



def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # savedir = f"samples/{Path(args.config).stem}-{time_str}"
    # os.makedirs(savedir)

    # clip model to get similarity scores
    # rank=torch.multiprocessing.current_process()._identity[0]-1
    clip_model, preprocess = clip.load("ViT-L/14")

    config  = OmegaConf.load(args.config)
    # samples = []
    
    # sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            # tokenizer_path = os.path.join(args.pretrained_model_path, "tokenizer/vocab.json")
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            # tokenizer    = CLIPTokenizer.from_pretrained(tokenizer_path)
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            # random_seeds = model_config.get("seed", [-1])
            # random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            # random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            # for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            num_prompts = 100 #470 # 150
            n_prompts = n_prompts * num_prompts
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * num_prompts if len(random_seeds) == 1 else random_seeds
            for object_metadata in YTVIS_CATEGORIES_2021:
                object_class = object_metadata["name"]
                object_id = object_metadata["id"]
                if int(object_id) < 25:
                    continue
                # prompts = make_prompts(object_metadata, num_prompts)

                # prompts = [f"a close up of running {object_class}, centred", f"a close up of a flying {object_class}, centred"]
                # prompts = [f"a close up of one moving dynamic {object_class} in changing background, moving camera, centred"]

                # prompts = [f"a scene with smoke rising in forest, bushfire, small"]
                prompts = [f"a close up scene of rising smoke from forest bushfire, centered"]
                # prompts = [f"a close up scene of rising smoke, centered"]
                # num_prompts = 10
                prompts = [p for p in prompts for _ in range(num_prompts)]
                # print(f'prompts: {prompts}')
                # class_dir = f"VXpaste/youtube_vis_470/{object_class}"
                class_dir = f"Bushfire/gen_images3"

                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                    # if prompt_idx < 61:
                    #     continue
                    print(f'prompt idx: {prompt_idx} of {len(prompts)}')
                    savedir = os.path.join(class_dir, f'{prompt_idx}')
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    # prompt = f"a close up of running {object_class}, centred"
                    # manually set random seed for reproduction
                    if random_seed != -1: torch.manual_seed(random_seed)
                    else: torch.seed()
                    config[config_key].random_seed.append(torch.initial_seed())

                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    sample = pipeline(
                        prompt,
                        negative_prompt     = n_prompt,
                        num_inference_steps = model_config.steps,
                        guidance_scale      = model_config.guidance_scale,
                        width               = args.W,
                        height              = args.H,
                        video_length        = args.L,
                    ).videos
                    # samples.append(sample)
                    # exit()
                    clips = get_clip_scores(sample, prompt, clip_model, preprocess)

                    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    # obj_class = "ape"
                    save_videos_frames(sample, savedir, object_class)
                    print(f"save to {savedir}")
                    # save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                    # print(f"save to {savedir}/sample/{prompt}.gif")

                    # for each clip, save a json file with clip scores
                    save_dict={}
                    save_dict["name"] = object_class
                    save_dict["prompt"] = prompt
                    save_dict["id"] = object_id
                    save_dict["seq_id"] = prompt_idx
                    # for i, j in zip(target_class, results):
                    save_dict['clip_scores'] = clips
                    with open(os.path.join(savedir, "results.json"), 'w') as f:
                        json.dump(save_dict, f)

                # gc.collect()
                    # sample_idx += 1
                exit()

    # samples = torch.concat(samples)
    # save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
