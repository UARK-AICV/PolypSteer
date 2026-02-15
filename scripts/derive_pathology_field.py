from __future__ import annotations

import argparse
import json

import torch
from huggingface_hub import snapshot_download

from medsteer import (
    DEFAULT_LAYER_END,
    DEFAULT_LAYER_START,
    DEFAULT_NUM_STEPS,
    DEFAULT_NUM_VECTOR_SEEDS,
    PromptPair,
    SSPSProtocol,
    derive_pathology_field,
    fixed_prompt_pairs,
    load_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="The PolypSteer pathology-field derivation protocol."
    )
    parser.add_argument("--model", default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--lora_repo", default="phamtrongthang/medsteer")
    parser.add_argument(
        "--positive_prompt",
        default="An endoscopic image of dyed lifted polyps",
    )
    parser.add_argument(
        "--negative_prompt",
        default="An endoscopic image of polyps",
    )
    parser.add_argument("--prompt_pairs_jsonl", default=None)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_NUM_VECTOR_SEEDS)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--layer_start", type=int, default=DEFAULT_LAYER_START)
    parser.add_argument("--layer_end", type=int, default=DEFAULT_LAYER_END)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--memory_mode",
        choices=["auto", "resident", "model_offload", "sequential_offload"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        default="outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz",
    )
    return parser.parse_args()


def load_pairs(args):
    if args.prompt_pairs_jsonl is None:
        seeds = range(args.seed_start, args.seed_start + args.num_seeds)
        return fixed_prompt_pairs(args.positive_prompt, args.negative_prompt, seeds)
    pairs = []
    with open(args.prompt_pairs_jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pairs.append(
                PromptPair(
                    positive=record["positive_prompt"],
                    negative=record["negative_prompt"],
                    seed=int(record["seed"]),
                )
            )
    return tuple(pairs)


def main():
    args = parse_args()
    checkpoint = args.lora_path or snapshot_download(args.lora_repo)
    pipe = load_pipeline(
        model_id=args.model,
        lora_path=checkpoint,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        memory_mode=args.memory_mode,
    )
    protocol = SSPSProtocol(
        layer_ids=tuple(range(args.layer_start, args.layer_end)),
        num_steps=args.num_steps,
    )
    derive_pathology_field(
        pipe,
        load_pairs(args),
        protocol=protocol,
        batch_size=args.batch_size,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
