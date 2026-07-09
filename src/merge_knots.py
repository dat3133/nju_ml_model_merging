from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LORA_A_RE = re.compile(r"(.+)\.lora_A(?:\.[^.]+)?\.weight$")
LORA_B_RE = re.compile(r"(.+)\.lora_B(?:\.[^.]+)?\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KnOTS-style SVD projection plus TIES merge for two or more LoRA adapters."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--svd-rank", type=int, default=16, help="0 means keep full rank.")
    parser.add_argument("--majority-sign-method", choices=["total", "frequency"], default="total")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf-cache")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(dtype_name)


def load_adapter_state(adapter_path: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    path = Path(adapter_path)
    safe_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safe_path.exists():
        return load_file(str(safe_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin under {adapter_path}")


def lora_prefixes(state: dict[str, Any]) -> dict[str, tuple[str, str]]:
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for key in state:
        a_match = LORA_A_RE.match(key)
        if a_match:
            a_keys[a_match.group(1)] = key
            continue
        b_match = LORA_B_RE.match(key)
        if b_match:
            b_keys[b_match.group(1)] = key
    prefixes: dict[str, tuple[str, str]] = {}
    for prefix, a_key in a_keys.items():
        if prefix in b_keys:
            prefixes[prefix] = (a_key, b_keys[prefix])
    return prefixes


def base_weight_key(prefix: str) -> str:
    for start in ("base_model.model.", "base_model."):
        if prefix.startswith(start):
            prefix = prefix[len(start) :]
            break
    return f"{prefix}.weight"


def first_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        for item in value.values():
            numeric = first_numeric(item)
            if numeric is not None:
                return numeric
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def load_lora_scale(adapter_path: str) -> float:
    config_path = Path(adapter_path) / "adapter_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    r = first_numeric(config.get("r")) or first_numeric(config.get("rank_pattern"))
    alpha = (
        first_numeric(config.get("lora_alpha"))
        or first_numeric(config.get("alpha"))
        or first_numeric(config.get("alpha_pattern"))
    )

    if r is None:
        try:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(adapter_path)
            r = first_numeric(getattr(peft_config, "r", None))
            alpha = alpha or first_numeric(getattr(peft_config, "lora_alpha", None))
        except Exception:
            pass

    if not r:
        raise ValueError(f"cannot infer LoRA rank for {adapter_path}")
    if alpha is None:
        alpha = r
    return float(alpha) / float(r)


def svd_project(delta: Any, rank: int):
    import torch

    if rank <= 0 or rank >= min(delta.shape):
        return delta
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def shared_right_basis_project(deltas: list[Any], rank: int):
    import torch

    if rank <= 0:
        return deltas
    full_rank = min(min(delta.shape) for delta in deltas)
    rank = min(rank, full_rank)
    concat = torch.cat([delta.float() for delta in deltas], dim=0)
    _, _, vh = torch.linalg.svd(concat, full_matrices=False)
    basis = vh[:rank, :].T
    return [(delta.float() @ basis) @ basis.T for delta in deltas]


def ties_merge_tensors(deltas: list[Any], weights: list[float], density: float, majority_sign_method: str):
    import torch

    weighted = torch.stack([delta.float() * float(weight) for delta, weight in zip(deltas, weights)])
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if density < 1.0:
        flat = weighted.abs().flatten(start_dim=1)
        k = max(1, int(flat.shape[1] * density))
        threshold = torch.topk(flat, k=k, dim=1).values[:, -1]
        keep = weighted.abs() >= threshold.view(-1, *([1] * (weighted.ndim - 1)))
        weighted = torch.where(keep, weighted, torch.zeros_like(weighted))

    if majority_sign_method == "frequency":
        sign = torch.sign(torch.sign(weighted).sum(dim=0))
    else:
        sign = torch.sign(weighted.sum(dim=0))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    mask = (torch.sign(weighted) == sign.unsqueeze(0)) & (weighted != 0)
    numerator = torch.where(mask, weighted, torch.zeros_like(weighted)).sum(dim=0)
    denominator = mask.sum(dim=0).clamp(min=1)
    return numerator / denominator


def main() -> None:
    args = parse_args()
    if args.weights is None:
        args.weights = [1.0] * len(args.adapters)
    if len(args.weights) != len(args.adapters):
        raise ValueError("--weights and --adapters must have the same length")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_states = [load_adapter_state(path) for path in args.adapters]
    adapter_scales = [load_lora_scale(path) for path in args.adapters]
    prefix_sets = [set(lora_prefixes(state)) for state in adapter_states]
    common_prefixes = sorted(set.intersection(*prefix_sets))
    if not common_prefixes:
        raise ValueError("no common LoRA modules found across adapters")
    compute_device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=False,
    )
    if args.device != "cpu":
        model.to(compute_device)
    state = model.state_dict()

    applied: list[str] = []
    for prefix in common_prefixes:
        key = base_weight_key(prefix)
        if key not in state:
            print(f"skip {prefix}: base key {key} not found")
            continue
        deltas = []
        for adapter_state, scale in zip(adapter_states, adapter_scales):
            a_key, b_key = lora_prefixes(adapter_state)[prefix]
            a = adapter_state[a_key].to(device=compute_device, dtype=torch.float32)
            b = adapter_state[b_key].to(device=compute_device, dtype=torch.float32)
            delta = (b @ a) * scale
            delta = svd_project(delta, args.svd_rank)
            deltas.append(delta)
        deltas = shared_right_basis_project(deltas, args.svd_rank)
        merged_delta = ties_merge_tensors(deltas, args.weights, args.density, args.majority_sign_method)
        target = state[key]
        if tuple(target.shape) != tuple(merged_delta.shape):
            print(f"skip {prefix}: shape mismatch base={tuple(target.shape)} delta={tuple(merged_delta.shape)}")
            continue
        target.add_(merged_delta.to(device=target.device, dtype=target.dtype))
        applied.append(key)
        print(f"merged {key}", flush=True)
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(output)
    manifest = {
        "method": "knots_ties_lora_svd",
        "base_model": args.base_model,
        "adapters": args.adapters,
        "weights": args.weights,
        "density": args.density,
        "svd_rank": args.svd_rank,
        "majority_sign_method": args.majority_sign_method,
        "applied_base_weights": applied,
    }
    with open(output / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved KnOTS-style merged full model to {output}; applied {len(applied)} modules")


if __name__ == "__main__":
    main()
