# PolypSteer: Counterfactual Endoscopic Synthesis via Training-Free Activation Steering

[arXiv:2603.07066](https://arxiv.org/abs/2603.07066)

PolypSteer produces same-seed counterfactual endoscopy images by modifying selected PixArt-α cross-attention outputs during denoising. The implementation follows the Spatially Selective Pathology Steering protocol with a dedicated counterfactual sampler.

## SSPS protocol

The public reproduction uses `PixArt-alpha/PixArt-XL-2-512x512` with the LoRA checkpoint at `phamtrongthang/medsteer`. The pathology field is derived from 50 matched random seeds and the contrastive prompts:

```text
An endoscopic image of dyed lifted polyps
An endoscopic image of polyps
```

For every denoising step and selected transformer layer, the positive and negative cross-attention features are averaged over prompt pairs and spatial tokens. Their normalized difference is the pathology field:

$$
\mathbf v_{l,t}=
\frac{\bar{\mathbf h}^{+}_{l,t}-\bar{\mathbf h}^{-}_{l,t}}
{\lVert\bar{\mathbf h}^{+}_{l,t}-\bar{\mathbf h}^{-}_{l,t}\rVert_2}.
$$

The reported configuration selects PixArt layers 8–16 with width 8. The implementation therefore uses the half-open layer interval `[8, 16)`, corresponding to transformer layers 8 through 15. SSPS is applied at every selected layer and every denoising step:

$$
\sigma_{l,t}=\max(\langle\mathbf h_{l,t},\mathbf v_{l,t}\rangle,0),
\qquad
\mathbf h'_{l,t}=\mathbf h_{l,t}-2.5\,\sigma_{l,t}\mathbf v_{l,t}.
$$

The update has no cosine denominator and no post-update normalization. The sampler evaluates classifier-free guidance as separate unconditional and conditional transformer passes. Only the conditional cross-attention pass receives SSPS.

The unsteered and steered images use the same positive prompt and reconstructed generators with the same seed. The negative prompt is rendered only as a re-prompting reference and is not part of the counterfactual pair.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

```bash
pip install -e diffusers/
pip install -e ".[dev]"
```

## Public-checkpoint reproduction

The full command derives the 50-pair pathology field and renders seed 42:

```bash
bash reproduce.sh all
```

The two stages can also run separately:

```bash
bash reproduce.sh derive
bash reproduce.sh render
```

The derived field is saved as:

```text
outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz
```

The inference stage writes:

```text
outputs/counterfactual_pair/unsteered_seed42.png
outputs/counterfactual_pair/steered_seed42.png
outputs/counterfactual_pair/reprompted_reference_seed42.png
outputs/counterfactual_pair/comparison_seed42.png
outputs/counterfactual_pair/ssps_gate_statistics.json
```

The commands download `phamtrongthang/medsteer` through Hugging Face Hub unless `--lora_path` supplies a local checkpoint.

## Pathology-field derivation

```bash
python scripts/derive_pathology_field.py \
    --positive_prompt "An endoscopic image of dyed lifted polyps" \
    --negative_prompt "An endoscopic image of polyps" \
    --num_seeds 50 \
    --num_steps 20 \
    --layer_start 8 \
    --layer_end 16 \
    --output outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz
```

Context phrasing can vary across matched prompt pairs. A JSONL manifest supplies that protocol directly. Every record contains `positive_prompt`, `negative_prompt`, and `seed`:

```json
{"positive_prompt":"An endoscopic image of dyed lifted polyps","negative_prompt":"An endoscopic image of polyps","seed":0}
{"positive_prompt":"An endoscopic view of dyed lifted polyps","negative_prompt":"An endoscopic view of polyps","seed":1}
```

```bash
python scripts/derive_pathology_field.py \
    --prompt_pairs_jsonl prompt_pairs.jsonl \
    --output outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz
```

The NPZ artifact contains a dense float32 field with shape `[step, selected_layer, hidden_feature]`, explicit layer identifiers, both prompt sequences, and all seeds.

## Counterfactual synthesis

```bash
python scripts/render_counterfactual_pair.py \
    --pathology_field outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz \
    --positive_prompt "An endoscopic image of dyed lifted polyps" \
    --negative_prompt "An endoscopic image of polyps" \
    --seed 42 \
    --num_steps 20 \
    --layer_start 8 \
    --layer_end 16 \
    --strength 2.5 \
    --output_dir outputs/counterfactual_pair
```

## Python API

```python
from huggingface_hub import snapshot_download

from medsteer import SSPSProtocol, load_pipeline, synthesize

checkpoint = snapshot_download("phamtrongthang/medsteer")
pipeline = load_pipeline(lora_path=checkpoint)
protocol = SSPSProtocol(
    layer_ids=tuple(range(8, 16)),
    strength=2.5,
    num_steps=20,
)

unsteered = synthesize(
    pipeline,
    "An endoscopic image of dyed lifted polyps",
    seed=42,
    protocol=protocol,
)
steered = synthesize(
    pipeline,
    "An endoscopic image of dyed lifted polyps",
    seed=42,
    pathology_field="outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz",
    protocol=protocol,
)
```

## Repository layout

```text
medsteer/polypsteer.py                 SSPS protocol, sampler, and processor
scripts/derive_pathology_field.py      Matched-pair pathology-field derivation
scripts/render_counterfactual_pair.py  Same-seed counterfactual rendering
scripts/train.py                       PixArt LoRA training
scripts/train_val.py                   PixArt LoRA training with validation
reproduce.sh                           Public-checkpoint reproduction
```

The training commands default to LoRA rank 64.

## Verification

```bash
python -m compileall medsteer scripts
ruff check medsteer/polypsteer.py scripts/derive_pathology_field.py scripts/render_counterfactual_pair.py
ruff format --check medsteer/polypsteer.py scripts/derive_pathology_field.py scripts/render_counterfactual_pair.py
```

## License

The repository is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
