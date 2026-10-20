"""Launch dense Pi0.5 fine-tuning with an explicit parameter-freeze mode."""

from __future__ import annotations

import dataclasses
import os

import flax.nnx as nnx
import jax

import openpi.shared.nnx_utils as nnx_utils
import train as train_module


def _initialize_distributed_from_env() -> None:
    process_count = int(os.environ.get("JAX_PROCESS_COUNT", "1"))
    if process_count <= 1 or jax.distributed.is_initialized():
        return
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS", "").strip()
    process_id_text = os.environ.get("JAX_PROCESS_ID", "").strip()
    local_device_ids_text = os.environ.get("JAX_LOCAL_DEVICE_IDS", "").strip()
    missing = [
        name
        for name, value in (
            ("JAX_COORDINATOR_ADDRESS", coordinator_address),
            ("JAX_PROCESS_ID", process_id_text),
            ("JAX_LOCAL_DEVICE_IDS", local_device_ids_text),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Multi-host JAX configuration is incomplete: " + ", ".join(missing)
        )
    local_device_ids = [
        int(value) for value in local_device_ids_text.split(",") if value.strip()
    ]
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=process_count,
        process_id=int(process_id_text),
        local_device_ids=local_device_ids,
        initialization_timeout=600,
    )


def _freeze_filter(mode: str) -> nnx.filterlib.Filter:
    vision_encoder = nnx_utils.PathRegex(".*PaliGemma.*img.*")
    vlm_without_action_expert = nnx.All(
        nnx_utils.PathRegex(".*PaliGemma.*llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    )

    if mode == "expert_only":
        # Freeze SigLIP and the Gemma 2B VLM. The Gemma 300M action expert,
        # action input/output projections, and time MLP remain trainable.
        return nnx.Any(vision_encoder, vlm_without_action_expert)
    if mode == "freeze_vision":
        # Freeze only SigLIP; train both language/action transformers and heads.
        return vision_encoder
    if mode == "full":
        # Freeze nothing: SigLIP, Gemma 2B, the Gemma 300M action expert, and
        # all action/time projections are optimized together.
        return nnx.Nothing
    raise ValueError(
        f"Unsupported PI05_FINETUNE_MODE={mode!r}; expected expert_only, "
        "freeze_vision, or full"
    )


def main() -> None:
    _initialize_distributed_from_env()
    config = train_module._config.cli()
    local_device_count = jax.local_device_count()
    expected_local_device_count = int(
        os.environ.get("PI05_EXPECTED_LOCAL_GPUS", "8")
    )
    if local_device_count != expected_local_device_count:
        raise RuntimeError(
            f"Expected {expected_local_device_count} local GPUs, "
            f"found {local_device_count}"
        )
    global_device_count = jax.device_count()
    fsdp_devices = int(
        os.environ.get("PI05_FSDP_DEVICES", str(local_device_count))
    )

    mode = os.environ.get("PI05_FINETUNE_MODE", "").strip()
    freeze_filter = _freeze_filter(mode)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        freeze_filter=freeze_filter,
        fsdp_devices=fsdp_devices,
    )
    print(
        "JAX_TOPOLOGY="
        f"process={jax.process_index()}/{jax.process_count()} "
        f"local_devices={local_device_count} "
        f"global_devices={global_device_count} "
        f"fsdp_devices={fsdp_devices}",
        flush=True,
    )
    print(f"PI05_FINETUNE_MODE={mode}", flush=True)
    print(f"PI05_FREEZE_FILTER={freeze_filter!r}", flush=True)
    train_module.main(config)


if __name__ == "__main__":
    main()
