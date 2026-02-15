from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import numpy as np

DEFAULT_LAYER_START = 8
DEFAULT_LAYER_END = 16
DEFAULT_NUM_STEPS = 20
DEFAULT_NUM_VECTOR_SEEDS = 50
DEFAULT_STEERING_STRENGTH = 2.5
PATHOLOGY_FIELD_FORMAT = "polypsteer-pathology-field-v3"


@dataclass(frozen=True, slots=True)
class SSPSProtocol:
    layer_ids: tuple[int, ...] = tuple(range(DEFAULT_LAYER_START, DEFAULT_LAYER_END))
    strength: float = DEFAULT_STEERING_STRENGTH
    num_steps: int = DEFAULT_NUM_STEPS

    def validate(self, total_layers: int) -> None:
        if not self.layer_ids:
            raise ValueError("The layer selection is empty.")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("The layer selection contains duplicates.")
        if min(self.layer_ids) < 0 or max(self.layer_ids) >= total_layers:
            raise ValueError("The layer selection is outside the PixArt transformer.")
        if self.strength < 0:
            raise ValueError("The steering strength is negative.")
        if self.num_steps < 1:
            raise ValueError("The denoising step count is invalid.")


@dataclass(frozen=True, slots=True)
class PromptPair:
    positive: str
    negative: str
    seed: int


@dataclass(frozen=True, slots=True)
class PathologyField:
    values: np.ndarray
    layer_ids: tuple[int, ...]
    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]
    seeds: tuple[int, ...]

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                "The pathology field must have step, layer, and feature axes."
            )
        if self.values.shape[1] != len(self.layer_ids):
            raise ValueError(
                "The pathology field and layer identifiers differ in width."
            )
        if not np.isfinite(self.values).all():
            raise ValueError("The pathology field contains non-finite values.")
        if np.any(np.linalg.norm(self.values, axis=-1) <= 1e-8):
            raise ValueError("The pathology field contains a zero direction.")
        count = len(self.seeds)
        if len(self.positive_prompts) != count or len(self.negative_prompts) != count:
            raise ValueError("The prompt-pair metadata is incomplete.")


@dataclass(frozen=True, slots=True)
class CrossAttentionTrace:
    values: np.ndarray
    layer_ids: tuple[int, ...]


def save_pathology_field(field: PathologyField, path: str | os.PathLike) -> Path:
    field.validate()
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray(PATHOLOGY_FIELD_FORMAT),
        values=field.values.astype(np.float32),
        layer_ids=np.asarray(field.layer_ids, dtype=np.int64),
        positive_prompts=np.asarray(field.positive_prompts),
        negative_prompts=np.asarray(field.negative_prompts),
        seeds=np.asarray(field.seeds, dtype=np.int64),
    )
    return output


def load_pathology_field(
    source: PathologyField | str | os.PathLike,
) -> PathologyField:
    if isinstance(source, PathologyField):
        source.validate()
        return source
    with np.load(source) as archive:
        if str(archive["format"].item()) != PATHOLOGY_FIELD_FORMAT:
            raise ValueError(f"The pathology field format in {source} is unsupported.")
        field = PathologyField(
            values=np.asarray(archive["values"], dtype=np.float32),
            layer_ids=tuple(int(value) for value in archive["layer_ids"].tolist()),
            positive_prompts=tuple(
                str(value) for value in archive["positive_prompts"].tolist()
            ),
            negative_prompts=tuple(
                str(value) for value in archive["negative_prompts"].tolist()
            ),
            seeds=tuple(int(value) for value in archive["seeds"].tolist()),
        )
    field.validate()
    return field


def load_pipeline(
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
    lora_path: str | os.PathLike | None = None,
    device: str | None = None,
    dtype=None,
    memory_mode: str = "auto",
):
    import torch
    from peft import PeftModel
    from transformers import T5EncoderModel

    from diffusers import PixArtAlphaPipeline, Transformer2DModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    if memory_mode not in {
        "auto",
        "resident",
        "model_offload",
        "sequential_offload",
    }:
        raise ValueError(f"The memory mode {memory_mode} is unsupported.")

    variant = "fp16" if dtype == torch.float16 else None
    transformer = Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    text_options = {"subfolder": "text_encoder", "torch_dtype": dtype}
    if variant:
        text_options["variant"] = variant
    text_encoder = T5EncoderModel.from_pretrained(model_id, **text_options)

    if lora_path:
        checkpoint = Path(lora_path)
        transformer = PeftModel.from_pretrained(
            transformer, str(checkpoint / "transformer_lora")
        )
        text_encoder = PeftModel.from_pretrained(
            text_encoder, str(checkpoint / "text_encoder_lora")
        )

    pipeline_options = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "torch_dtype": dtype,
    }
    if variant:
        pipeline_options["variant"] = variant
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, **pipeline_options)

    if memory_mode == "auto":
        if device == "cuda":
            total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_mode = "sequential_offload" if total_gib < 16 else "resident"
        else:
            memory_mode = "resident"

    if device == "cuda" and memory_mode == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    elif device == "cuda" and memory_mode == "model_offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if device == "cuda" and find_spec("xformers") is not None:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except (AttributeError, ImportError, RuntimeError, ValueError):
            pass
    return pipe


class _SSPSProcessor:
    def __init__(self, original, session, slot: int):
        self.original = original
        self.session = session
        self.slot = slot

    def __call__(
        self,
        attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        output = self.original(
            attention,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            *args,
            **kwargs,
        )
        return self.session.apply(self.slot, output)


class _SSPSSession(AbstractContextManager):
    def __init__(
        self,
        transformer,
        protocol: SSPSProtocol,
        field: PathologyField | None,
        collect: bool,
        keep_gate_statistics: bool,
    ):
        blocks = list(transformer.transformer_blocks)
        protocol.validate(len(blocks))
        if field is not None:
            field.validate()
            if field.layer_ids != protocol.layer_ids:
                raise ValueError(
                    "The pathology field uses a different layer selection."
                )
            if field.num_steps != protocol.num_steps:
                raise ValueError("The pathology field uses a different step count.")
        self.protocol = protocol
        self.field = field
        self.collect = collect
        self.keep_gate_statistics = keep_gate_statistics
        self.modules = [blocks[index].attn2 for index in protocol.layer_ids]
        self.originals = []
        self.step = None
        self.seen = np.zeros((protocol.num_steps, len(self.modules)), dtype=bool)
        self.trace = None
        shape = (protocol.num_steps, len(self.modules))
        self.gate_mean = np.zeros(shape, dtype=np.float32)
        self.gate_max = np.zeros(shape, dtype=np.float32)
        self.gate_fraction = np.zeros(shape, dtype=np.float32)

    def __enter__(self):
        for slot, module in enumerate(self.modules):
            original = module.processor
            self.originals.append((module, original))
            module.set_processor(_SSPSProcessor(original, self, slot))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, original in self.originals:
            module.set_processor(original)
        self.originals.clear()
        return False

    @contextmanager
    def denoising_step(self, step: int | None):
        previous = self.step
        self.step = step
        try:
            yield
        finally:
            self.step = previous

    def apply(self, slot: int, output):
        import torch

        if self.step is None:
            return output
        step = self.step
        if self.seen[step, slot]:
            raise RuntimeError(
                "A selected cross-attention layer ran twice in one step."
            )
        self.seen[step, slot] = True

        if self.collect:
            summary = output.detach().float().mean(dim=1).cpu().numpy()
            if self.trace is None:
                self.trace = np.empty(
                    (
                        self.protocol.num_steps,
                        len(self.modules),
                        summary.shape[0],
                        summary.shape[1],
                    ),
                    dtype=np.float32,
                )
            self.trace[step, slot] = summary

        if self.field is None:
            return output

        vector = torch.as_tensor(
            self.field.values[step, slot],
            device=output.device,
            dtype=output.dtype,
        ).reshape(1, 1, -1)
        alignment = (output * vector).sum(dim=-1, keepdim=True).clamp_min(0)
        revised = output - self.protocol.strength * alignment * vector
        if self.keep_gate_statistics:
            values = alignment.detach().float()
            self.gate_mean[step, slot] = values.mean().item()
            self.gate_max[step, slot] = values.max().item()
            self.gate_fraction[step, slot] = (values > 0).float().mean().item()
        return revised

    def finish(self) -> None:
        if not self.seen.all():
            missing = np.argwhere(~self.seen).tolist()
            raise RuntimeError(
                f"Selected cross-attention passes are missing: {missing}."
            )

    def cross_attention_trace(self) -> CrossAttentionTrace:
        self.finish()
        if self.trace is None:
            raise RuntimeError("The session did not collect cross-attention features.")
        return CrossAttentionTrace(self.trace, self.protocol.layer_ids)

    def gate_statistics(self) -> dict[str, np.ndarray]:
        self.finish()
        return {
            "mean": self.gate_mean.copy(),
            "max": self.gate_max.copy(),
            "active_fraction": self.gate_fraction.copy(),
        }


def _execution_device(pipe):
    device = getattr(pipe, "_execution_device", None)
    if device is not None:
        return device
    return next(pipe.transformer.parameters()).device


def _timestep_batch(timestep, batch: int, device):
    import torch

    if not torch.is_tensor(timestep):
        dtype = torch.float64 if isinstance(timestep, float) else torch.int64
        timestep = torch.tensor([timestep], dtype=dtype, device=device)
    elif timestep.ndim == 0:
        timestep = timestep[None].to(device)
    return timestep.expand(batch)


def _micro_conditions(pipe, batch: int, dtype, device):
    import torch

    if pipe.transformer.config.sample_size != 128:
        return {"resolution": None, "aspect_ratio": None}
    resolution = torch.tensor([1024, 1024], device=device, dtype=dtype).repeat(batch, 1)
    aspect_ratio = torch.ones((batch, 1), device=device, dtype=dtype)
    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def _sample(
    pipe,
    prompts: Sequence[str],
    seeds: Sequence[int],
    protocol: SSPSProtocol,
    session: _SSPSSession | None,
    output_type: str,
):
    import torch

    if len(prompts) != len(seeds) or not prompts:
        raise ValueError("Prompts and seeds must form a nonempty matched sequence.")
    device = _execution_device(pipe)
    batch = len(prompts)
    negatives = [""] * batch
    (
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
    ) = pipe.encode_prompt(
        list(prompts),
        True,
        negative_prompt=negatives,
        num_images_per_prompt=1,
        device=device,
        clean_caption=True,
        max_sequence_length=120,
    )
    pipe.scheduler.set_timesteps(protocol.num_steps, device=device)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    latent_channels = pipe.transformer.config.in_channels
    size = pipe.transformer.config.sample_size * pipe.vae_scale_factor
    latents = pipe.prepare_latents(
        batch,
        latent_channels,
        size,
        size,
        prompt_embeds.dtype,
        device,
        generators,
        None,
    )
    step_options = pipe.prepare_extra_step_kwargs(generators, 0.0)
    conditions = _micro_conditions(pipe, batch, prompt_embeds.dtype, device)

    with torch.no_grad():
        for step, timestep in enumerate(pipe.scheduler.timesteps):
            model_input = pipe.scheduler.scale_model_input(latents, timestep)
            timestep_input = _timestep_batch(timestep, batch, model_input.device)
            context = session.denoising_step(None) if session else nullcontext()
            with context:
                unconditional = pipe.transformer(
                    model_input,
                    encoder_hidden_states=negative_embeds,
                    encoder_attention_mask=negative_mask,
                    timestep=timestep_input,
                    added_cond_kwargs=conditions,
                    return_dict=False,
                )[0]
            context = session.denoising_step(step) if session else nullcontext()
            with context:
                conditional = pipe.transformer(
                    model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timestep_input,
                    added_cond_kwargs=conditions,
                    return_dict=False,
                )[0]
            prediction = unconditional + 4.5 * (conditional - unconditional)
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                prediction = prediction.chunk(2, dim=1)[0]
            latents = pipe.scheduler.step(
                prediction,
                timestep,
                latents,
                **step_options,
                return_dict=False,
            )[0]

    if output_type == "latent":
        return latents
    with torch.no_grad():
        image = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
    return pipe.image_processor.postprocess(image, output_type=output_type)


def synthesize(
    pipe,
    prompt: str,
    *,
    seed: int = 0,
    pathology_field: PathologyField | str | os.PathLike | None = None,
    protocol: SSPSProtocol | None = None,
    keep_gate_statistics: bool = False,
):
    field = (
        load_pathology_field(pathology_field) if pathology_field is not None else None
    )
    if protocol is None:
        protocol = SSPSProtocol(
            layer_ids=field.layer_ids if field is not None else tuple(range(8, 16)),
            num_steps=field.num_steps if field is not None else DEFAULT_NUM_STEPS,
        )
    if field is None:
        return _sample(pipe, [prompt], [seed], protocol, None, "pil")[0]
    session = _SSPSSession(
        pipe.transformer,
        protocol,
        field,
        collect=False,
        keep_gate_statistics=keep_gate_statistics,
    )
    with session:
        image = _sample(pipe, [prompt], [seed], protocol, session, "pil")[0]
    session.finish()
    if keep_gate_statistics:
        return image, session.gate_statistics()
    return image


def _trace_prompt_pairs(
    pipe,
    prompts: Sequence[str],
    seeds: Sequence[int],
    protocol: SSPSProtocol,
) -> CrossAttentionTrace:
    session = _SSPSSession(
        pipe.transformer,
        protocol,
        field=None,
        collect=True,
        keep_gate_statistics=False,
    )
    with session:
        _sample(pipe, prompts, seeds, protocol, session, "latent")
    return session.cross_attention_trace()


def derive_pathology_field(
    pipe,
    prompt_pairs: Iterable[PromptPair],
    *,
    protocol: SSPSProtocol | None = None,
    batch_size: int = 1,
    save_path: str | os.PathLike | None = None,
) -> PathologyField:
    pairs = tuple(prompt_pairs)
    if not pairs:
        raise ValueError("At least one contrastive prompt pair is required.")
    if batch_size < 1:
        raise ValueError("The batch size must be positive.")
    protocol = protocol or SSPSProtocol()
    total = None
    completed = 0
    for offset in range(0, len(pairs), batch_size):
        group = pairs[offset : offset + batch_size]
        seeds = [pair.seed for pair in group]
        positive = _trace_prompt_pairs(
            pipe,
            [pair.positive for pair in group],
            seeds,
            protocol,
        ).values
        negative = _trace_prompt_pairs(
            pipe,
            [pair.negative for pair in group],
            seeds,
            protocol,
        ).values
        delta = (positive - negative).sum(axis=2, dtype=np.float64)
        total = delta if total is None else total + delta
        completed += len(group)
        print(f"Pathology field pairs: {completed}/{len(pairs)}", flush=True)

    mean_delta = total / len(pairs)
    norms = np.linalg.norm(mean_delta, axis=-1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("At least one pathology direction is degenerate.")
    field = PathologyField(
        values=(mean_delta / norms).astype(np.float32),
        layer_ids=protocol.layer_ids,
        positive_prompts=tuple(pair.positive for pair in pairs),
        negative_prompts=tuple(pair.negative for pair in pairs),
        seeds=tuple(pair.seed for pair in pairs),
    )
    if save_path is not None:
        save_pathology_field(field, save_path)
    return field


def fixed_prompt_pairs(
    positive: str,
    negative: str,
    seeds: Iterable[int] = range(DEFAULT_NUM_VECTOR_SEEDS),
) -> tuple[PromptPair, ...]:
    return tuple(PromptPair(positive, negative, int(seed)) for seed in seeds)
