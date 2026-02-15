"""
Batch image generation from metadata.csv prompts.

Generates images using PixArt-Alpha with LoRA adapters, reading prompts
from a metadata CSV file.
"""

import argparse
import gc
import os

import pandas as pd
import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel
from tqdm import tqdm
from transformers import T5EncoderModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch image generation from PixArt-Alpha with LoRA."
    )
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset directory containing metadata.csv")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional path to metadata.csv")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="LoRA checkpoint path")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading model from {args.model_id} and LoRA from {args.ckpt_path}...")

    transformer = Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float32
    )
    transformer = PeftModel.from_pretrained(
        transformer, os.path.join(args.ckpt_path, "transformer_lora")
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.float32
    )
    text_encoder = PeftModel.from_pretrained(
        text_encoder, os.path.join(args.ckpt_path, "text_encoder_lora")
    )

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.float32,
    )
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Could not enable xformers: {e}. Falling back to default attention.")

    if args.csv_path and os.path.exists(args.csv_path):
        csv_path = args.csv_path
    else:
        csv_path = os.path.join(args.dataset_id, "metadata.csv")
        if not os.path.exists(csv_path):
            if os.path.exists(os.path.join(args.dataset_id, "train", "metadata.csv")):
                csv_path = os.path.join(args.dataset_id, "train", "metadata.csv")
            elif os.path.exists(os.path.join(args.dataset_id, "test", "metadata.csv")):
                csv_path = os.path.join(args.dataset_id, "test", "metadata.csv")
            else:
                raise FileNotFoundError(
                    f"metadata.csv not found in {args.dataset_id} or its train/test subfolders."
                )

    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    base_seed = args.seed if args.seed is not None else 0

    for i in tqdm(range(0, args.num_images, args.batch_size), desc="Generating images"):
        current_batch_size = min(args.batch_size, args.num_images - i)

        batch_prompts = []
        batch_file_names = []
        batch_seeds = []

        for j in range(current_batch_size):
            idx = i + j
            row = df.iloc[idx % len(df)]
            batch_prompts.append(row["text"])
            batch_file_names.append(row["file_name"])
            batch_seeds.append(base_seed + idx)

        # Skip batch if all images already exist
        batch_output_paths = []
        all_exist = True
        for file_name, seed in zip(batch_file_names, batch_seeds):
            base_name, extension = os.path.splitext(file_name)
            output_path = os.path.join(args.save_dir, f"{base_name}_{seed}{extension}")
            batch_output_paths.append(output_path)
            if not os.path.exists(output_path):
                all_exist = False

        if all_exist:
            continue

        generators = [torch.Generator("cuda").manual_seed(s) for s in batch_seeds]

        with torch.no_grad():
            output = pipe(
                batch_prompts,
                num_inference_steps=args.num_inference_steps,
                generator=generators,
                use_resolution_binning=False,
            )

            for j, generated_image in enumerate(output.images):
                generated_image.save(batch_output_paths[j])

        del output
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
