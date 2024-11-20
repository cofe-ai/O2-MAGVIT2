# -*- coding:UTF-8 -*-
import torch
import sys
import os
from omegaconf import OmegaConf
from modeling.magvit_model import VisionTokenizer
from torchvision.io import read_video, write_video, read_image, write_png
from torchvision.io import ImageReadMode
import torchvision.transforms as T
from einops import rearrange
import time
import pathlib
import random
from collections import OrderedDict
from src.utils import get_config, preprocess_vision_input, calculate_resize_shape
from PIL import Image
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--modal", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--target-pixels", type=int, default=384, help="target pixels for the short edge")
    parser.add_argument("--max-num-frames", required=False, default=None, type=int)
    parser.add_argument("--extract-frame-interval", required=False, default=1, type=int)

    parser.add_argument("-i", type=str, required=True, help="input image/video path")
    parser.add_argument("-o", type=str, required=True, help="output dir")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_path = args.i
    output_dir = args.o
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    states = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model_config = get_config(args.config_path)
    model = VisionTokenizer(config=model_config, commitment_cost=0, diversity_gamma=0, use_gan=False, use_lecam_ema=False, use_perceptual=False)
    model.tokenizer.load_state_dict(states, strict=True)
    tokenizer = model.tokenizer
    modal = args.modal
    model.eval()
    model.to("cuda")
    spatial_downsample_ratio = 2 ** (len(model_config.model.decoder.channel_multipliers) - 1)
    temporal_downsample_ratio = 2 ** (sum(model_config.model.decoder.temporal_downsample))

    with torch.no_grad():
        if modal == "image":
            img = Image.open(input_path).convert("RGB")
            resize_shape = calculate_resize_shape(source_image_size=img.size, target_pixels=args.target_pixels, min_square_size=spatial_downsample_ratio)
            print(f"original image size: {img.size[::-1]} | resize shape: {resize_shape}")
            frames_torch = T.functional.pil_to_tensor(img).unsqueeze(dim=0)
            frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
            frames = frames.unsqueeze(dim=2)
        elif modal == "video":
            frames_torch = read_video(input_path, output_format="TCHW", pts_unit="sec")[0]
            original_shape = frames_torch.shape
            orig_img_h, orig_img_w = frames_torch.shape[-2:]
            frames_torch = frames_torch[::args.extract_frame_interval][:args.max_num_frames]
            resize_shape = calculate_resize_shape(source_image_size=(orig_img_w, orig_img_h), target_pixels=args.target_pixels, min_square_size=spatial_downsample_ratio)
            frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
            print(f"original video shape: {original_shape} | resize shape: {resize_shape}")
            frames = frames.unsqueeze(dim=0).permute(0, 2, 1, 3, 4) # n c t h w
        h, w = frames.shape[-2:]
        h = h // spatial_downsample_ratio
        w = w // spatial_downsample_ratio

        frames = frames.to("cuda")
        s = time.time()
        _, encoded_output, *_ = tokenizer.encode(frames, entropy_loss_weight=0.0)
        e = time.time()
        print(f"=== encode cost: {e -s} s ===")
        token_ids = encoded_output.indices
        print(f"num tokens: {token_ids.size()}, uniques: {token_ids.unique().numel()}")
        s = time.time()
        quantized = tokenizer.quantize.indices_to_codes(indices=token_ids, project_out=True)

        quantized = rearrange(quantized, "b (t h w) c -> b c t h w", h=h, w=w)

        decoded_output = tokenizer.decode(quantized)
        e = time.time()
        print(f"=== decode cost: {e -s} s ===")
        decoded_output = decoded_output.detach()[0]
        if modal == "image":
            print(frames.squeeze(dim=0).squeeze(dim=1).shape, decoded_output.squeeze(dim=0).squeeze(dim=1).shape)
            write_png(((frames.squeeze(dim=0).squeeze(dim=1).detach().cpu()+1) * 127.5).to(torch.uint8), os.path.join(output_dir, "input.png"), compression_level=0)
            write_png(((decoded_output.squeeze(dim=0).squeeze(dim=1).detach().cpu().clamp(-1,1)+1) * 127.5).to(torch.uint8), os.path.join(output_dir, f"recon_output.png"), compression_level=0)
        elif modal == "video":
            write_video(os.path.join(output_dir, f"input.mp4"), ((rearrange(frames.squeeze(dim=0), "c t h w -> t h w c")+1) * 127.5).to(torch.uint8), fps=10)
            write_video(os.path.join(output_dir, f"recon_output.mp4"), ((rearrange(decoded_output, "c t h w -> t h w c").clamp(-1,1)+1) * 127.5).to(torch.uint8), fps=10)

if __name__ == "__main__":
    main()