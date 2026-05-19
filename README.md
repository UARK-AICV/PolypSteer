# PolypSteer: Counterfactual Endoscopic Synthesis via Training-Free Activation Steering

**PolypSteer** is a training-free method for steering a fine-tuned Diffusion Transformer (DiT) at inference time, enabling controllable counterfactual synthesis of endoscopic images.
Given a base model fine-tuned on the [Kvasir](https://datasets.simula.no/kvasir/) endoscopy dataset, PolypSteer intercepts cross-attention activations inside the transformer blocks and shifts them along concept directions computed from cached activations — with no additional training required.

---

## Overview

Medical image generation benefits from the ability to synthesize counterfactual examples: images that are realistic but reflect a different clinical finding than the original. PolypSteer achieves this by:

1. **Capturing** cross-attention activations from the DiT when generating real and pathological prompts.
2. **Computing** a direction vector per denoising step and transformer block as the mean difference between two concept classes.
3. **Steering** new generations by suppressing these directions inside the frozen model at inference time.

The base generative model is [PixArt-α (PixArt-XL-2-512x512)](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512), fine-tuned via LoRA on Kvasir. Steering is applied to cross-attention output activations in all 28 `BasicTransformerBlock`s.

---

## Repository Structure

```
PolypSteer/
├── PolypSteer/                  # Core library
│   ├── capture.py             # ActivationRecorder — records activations per step/block
│   ├── directions.py          # compute_directions — mean-difference direction vectors
│   ├── hooks.py               # attach_hooks — monkey-patches DiT transformer blocks
│   ├── modulator.py           # AttentionModulator — suppress / record logic
│   ├── pipeline.py            # PolypSteerPipeline — high-level generation API
│   ├── losses.py              # color_distribution_loss — color consistency loss for LoRA
│   ├── classifier/            # KvasirClassifier (timm ConvNeXt-L, 8-class)
│   ├── evaluation/            # FID computation, classifier-based evaluation
│   └── visualization/         # Comparison grid generation (baseline / steered / delta)
├── scripts/
│   ├── train.py               # LoRA fine-tuning of PixArt-α on Kvasir
│   ├── train_val.py           # LoRA training with classifier-guided validation
│   ├── capture_activations.py # Record activations for the direction dataset
│   ├── compute_directions.py  # Compute mean-difference direction vectors
│   ├── generate.py            # Inference — baseline / suppress
│   ├── sample_batch.py        # Batch generation without steering
│   ├── evaluate_fid.py        # FID evaluation
│   └── evaluate_classifier.py # Classifier accuracy / F1 / AUC evaluation
├── configs/
│   └── accelerate_config.yaml # Multi-GPU Accelerate configuration
├── run_steer.sh               # End-to-end demo script
├── test_PolypSteer.py           # Unit and integration tests
├── pyproject.toml
└── requirements.txt
```

---

## Method

### 1. LoRA Fine-Tuning

Fine-tune PixArt-α on the Kvasir dataset using LoRA adapters applied to both the transformer and the T5 text encoder. A `color_distribution_loss` is added to the standard diffusion loss to preserve realistic tissue colour statistics.

### 2. Activation Capture

During inference, the cross-attention output of every `BasicTransformerBlock` is recorded at every denoising step. For CFG-based generation the conditional half of the batch is captured.

### 3. Direction Computation

For a chosen concept pair (e.g. *dyed lifted polyps* vs. *normal cecum*), the direction vector at step $t$ and block $b$ is:

$$\mathbf{d}_{t,b} = \frac{\bar{\mathbf{a}}^+_{t,b} - \bar{\mathbf{a}}^-_{t,b}}{\|\bar{\mathbf{a}}^+_{t,b} - \bar{\mathbf{a}}^-_{t,b}\|_2}$$

where $\bar{\mathbf{a}}^+$ and $\bar{\mathbf{a}}^-$ are the mean activations for the positive and negative concept classes respectively.

### 4. Inference-Time Steering

Given a direction vector $\mathbf{d}_{t,b}$ and activation $\mathbf{a}$, the suppression operation removes the component along the concept direction:

$$\mathbf{a}' = \text{norm}\!\left(\mathbf{a} - \max(0,\,\mathbf{a}\cdot\mathbf{d}_{t,b})\,\mathbf{d}_{t,b}\right)$$

This operation preserves the original activation norm.

---

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, CUDA-capable GPU.

```bash
# Install the vendored diffusers fork
pip install -e diffusers/

# Install PolypSteer and all dependencies
pip install -e ".[dev]"
```

Optional but recommended for faster attention:

```bash
pip install xformers
```

---

## Notebook Demo

Run the cells below in Google Colab or any Jupyter environment (GPU recommended).

**Cell 1 — install** (run once; no restart needed)
```python
import subprocess, sys

subprocess.run(["git", "clone", "https://github.com/phamtrongthang123/PolypSteer"], check=True)

# Install the vendored diffusers fork without touching other packages
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "PolypSteer/diffusers/",
], check=True)

# Install PolypSteer + remaining deps (peft, timm, …)
# Pin transformers to a version that still exports FLAX_WEIGHTS_NAME
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.41.2,<4.46",
    "PolypSteer/",
], check=True)
```

**Cell 2 — load and generate**
```python
import torch

# Compatibility shim: newer transformers removed FLAX_WEIGHTS_NAME.
# Patch it back before any diffusers code is imported.
import transformers.utils as _tu
if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
    _tu.FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"

from huggingface_hub import snapshot_download
from PolypSteer import PolypSteerPipeline

# ── 1. Download the LoRA checkpoint from the Hub ──────────────────────────────
lora_path = snapshot_download(
    repo_id="phamtrongthang/PolypSteer",
    local_dir="PolypSteer_ckpt",
)
# lora_path now contains  transformer_lora/  and  text_encoder_lora/

# ── 2. Load the model ─────────────────────────────────────────────────────────
pipe = PolypSteerPipeline.from_pretrained(
    model_id="PixArt-alpha/PixArt-XL-2-512x512",
    lora_path=lora_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ── 3. Baseline generation ────────────────────────────────────────────────────
image = pipe.generate(
    prompt="An endoscopic image of dyed lifted polyps",
    seed=42,
    num_steps=20,
    mode="baseline",
)
image.save("baseline.png")
image  # display inline in Jupyter

# ── 4. Steered generation (suppress polyp features) ──────────────────────────
# Requires precomputed direction vectors (.pickle).
# Compute them with scripts/compute_directions.py after capturing activations,
# or download a precomputed file if provided in the repository.
#
# directions_path = "path/to/directions.pickle"
# steered = pipe.generate(
#     prompt="An endoscopic image of dyed lifted polyps",
#     seed=42,
#     mode="suppress",
#     direction_vectors_path=directions_path,
#     suppress_scale=2.0,
# )
# steered.save("steered.png")
```

> **Tip:** `suppress_scale` controls steering strength. Values 1–3 work well;
> start with `2.0` and increase to remove more pathological features.

---

## Usage

### Quick Demo

```bash
# Baseline generation
bash run_steer.sh baseline

# Suppress a pathological finding (e.g. steer a polyp image toward normal)
bash run_steer.sh suppress

# Run both
bash run_steer.sh all
```

Edit `run_steer.sh` to set `MODEL_PATH`, `LORA_PATH`, `DIRECTION_VECTORS`, and the concept pair.

---

### Step-by-Step Pipeline

#### Step 1 — Fine-tune PixArt-α on Kvasir

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_val.py \
    --pretrained_model_name_or_path PixArt-alpha/PixArt-XL-2-512x512 \
    --train_data_dir /path/to/kvasir \
    --output_dir outputs/lora_kvasir \
    --num_train_epochs 20 \
    --lora_rank 16 \
    --classifier_ckpt /path/to/classifier.ckpt
```

#### Step 2 — Capture Cross-Attention Activations

```bash
python scripts/capture_activations.py \
    --csv_path /path/to/kvasir/metadata.csv \
    --raw_csv_path /path/to/kvasir/raw_labels.csv \
    --save_dir outputs/activations \
    --ckpt_path outputs/lora_kvasir/checkpoint-best-acc \
    --num_images 200
```

For distributed capture across multiple GPUs:

```bash
torchrun --nproc_per_node 4 scripts/capture_activations.py \
    --rank $RANK --world_size 4 [...]
```

#### Step 3 — Compute Direction Vectors

```bash
python scripts/compute_directions.py \
    --activations_dir outputs/activations \
    --raw_csv_path /path/to/kvasir/raw_labels.csv \
    --save_dir outputs/directions \
    --concept_positive "dyed lifted polyps" \
    --concept_negative "normal cecum"
```

This produces a `.pickle` file with shape `(num_steps, num_blocks, 1152)`.

#### Step 4 — Generate Steered Images

```bash
# Suppress pathological features
python scripts/generate.py \
    --model PixArt-alpha/PixArt-XL-2-512x512 \
    --lora_path outputs/lora_kvasir/checkpoint-best-acc \
    --prompt "An endoscopic image of dyed lifted polyps" \
    --direction_vectors outputs/directions/PixArt-XL-2-512x512_dyed lifted polyps_normal cecum.pickle \
    --mode suppress \
    --suppress_scale 2.0 \
    --num_images 10 \
    --save_dir outputs/generated/suppress
```

#### Step 5 — Evaluate

**FID:**

```bash
python scripts/evaluate_fid.py \
    --gen_dir outputs/generated/suppress \
    --ref_dir /path/to/kvasir/images
```

**Classifier accuracy / F1 / AUC:**

```bash
python scripts/evaluate_classifier.py \
    --images_dir outputs/generated/suppress \
    --raw_csv_path /path/to/kvasir/raw_labels.csv \
    --classifier_ckpt /path/to/classifier.ckpt
```

---

### Programmatic API

```python
from PolypSteer import PolypSteerPipeline, load_directions

# Load model
pipe = PolypSteerPipeline.from_pretrained(
    model_id="PixArt-alpha/PixArt-XL-2-512x512",
    lora_path="outputs/lora_kvasir/checkpoint-best-acc",
    device="cuda",
)

# Load precomputed directions
directions = load_directions("outputs/directions/my_directions.pickle")

# Suppress pathological features (steer away from the concept)
image = pipe.generate(
    prompt="An endoscopic image of dyed lifted polyps",
    seed=42,
    mode="suppress",
    direction_vectors=directions,
    suppress_scale=2.0,
)
image.save("suppressed.png")
```

---

### Train the Kvasir Classifier

The classifier (ConvNeXt-L via `timm`) is used for classifier-guided training validation and for evaluating generated images.

```bash
python -m PolypSteer.classifier.train_classifier \
    --csv_path /path/to/kvasir/metadata.csv \
    --data_root /path/to/kvasir/images \
    --output_dir outputs/classifier \
    --model convnext_large \
    --epochs 50
```

---

## Tests

```bash
# CPU unit tests (fast)
python test_PolypSteer.py

# Full integration tests (requires GPU and model checkpoint)
python test_PolypSteer.py --gpu
```

Test coverage includes: `AttentionModulator` modes (passthrough, record, suppress), `color_distribution_loss`, direction vector save/load, FID computation, `FeatureStats`, and grid visualization utilities.

---

## Kvasir Classes

| Label | Description |
|---|---|
| `dyed lifted polyps` | Polyps after dye-spraying and lifting |
| `dyed resection margins` | Margins post-polypectomy with dye |
| `esophagitis` | Inflammation of the esophagus |
| `normal cecum` | Normal cecum |
| `normal pylorus` | Normal pylorus |
| `normal z-line` | Normal gastroesophageal junction |
| `polyps` | Colorectal polyps |
| `ulcerative colitis` | Ulcerative colitis |

---

## Citation


---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
Free for academic and non-commercial research use. **Commercial use is prohibited.**
