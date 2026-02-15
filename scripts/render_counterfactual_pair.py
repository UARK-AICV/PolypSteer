from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from medsteer import (
    DEFAULT_LAYER_END,
    DEFAULT_LAYER_START,
    DEFAULT_NUM_STEPS,
    DEFAULT_STEERING_STRENGTH,
    SSPSProtocol,
    load_pipeline,
    synthesize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="The PolypSteer same-seed counterfactual protocol."
    )
    parser.add_argument("--model", default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--lora_repo", default="phamtrongthang/medsteer")
    parser.add_argument("--pathology_field", required=True)
    parser.add_argument(
        "--positive_prompt",
        default="An endoscopic image of dyed lifted polyps",
    )
    parser.add_argument(
        "--negative_prompt",
        default="An endoscopic image of polyps",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--layer_start", type=int, default=DEFAULT_LAYER_START)
    parser.add_argument("--layer_end", type=int, default=DEFAULT_LAYER_END)
    parser.add_argument("--strength", type=float, default=DEFAULT_STEERING_STRENGTH)
    parser.add_argument(
        "--memory_mode",
        choices=["auto", "resident", "model_offload", "sequential_offload"],
        default="auto",
    )
    parser.add_argument("--output_dir", default="outputs/counterfactual_pair")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.lora_path or snapshot_download(args.lora_repo)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(
        model_id=args.model,
        lora_path=checkpoint,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        memory_mode=args.memory_mode,
    )
    protocol = SSPSProtocol(
        layer_ids=tuple(range(args.layer_start, args.layer_end)),
        strength=args.strength,
        num_steps=args.num_steps,
    )
    unsteered = synthesize(
        pipe,
        args.positive_prompt,
        seed=args.seed,
        protocol=protocol,
    )
    unsteered.save(output / f"unsteered_seed{args.seed}.png")
    reprompted = synthesize(
        pipe,
        args.negative_prompt,
        seed=args.seed,
        protocol=protocol,
    )
    reprompted.save(output / f"reprompted_reference_seed{args.seed}.png")
    steered, gates = synthesize(
        pipe,
        args.positive_prompt,
        seed=args.seed,
        pathology_field=args.pathology_field,
        protocol=protocol,
        keep_gate_statistics=True,
    )
    steered.save(output / f"steered_seed{args.seed}.png")
    gate_report = {
        metric: {
            str(step): {
                str(layer): float(values[step, slot])
                for slot, layer in enumerate(protocol.layer_ids)
            }
            for step in range(protocol.num_steps)
        }
        for metric, values in gates.items()
    }
    with open(output / "ssps_gate_statistics.json", "w", encoding="utf-8") as handle:
        json.dump(gate_report, handle, indent=2)
    comparison = np.concatenate(
        [np.asarray(unsteered), np.asarray(reprompted), np.asarray(steered)], axis=1
    )
    from PIL import Image

    Image.fromarray(comparison).save(output / f"comparison_seed{args.seed}.png")
    print(f"The counterfactual outputs are in {output}.")


if __name__ == "__main__":
    main()
