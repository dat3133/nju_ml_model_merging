from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapters with TIES or DARE-TIES and save a full model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--adapter-names", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--combination", choices=["ties", "dare_ties", "linear"], default="ties")
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--majority-sign-method", choices=["total", "frequency"], default="total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf-cache")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device-map", default="auto")
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


def call_add_weighted_adapter(model: Any, kwargs: dict[str, Any]) -> None:
    try:
        model.add_weighted_adapter(**kwargs)
    except TypeError:
        fallback = dict(kwargs)
        fallback.pop("majority_sign_method", None)
        model.add_weighted_adapter(**fallback)


def main() -> None:
    args = parse_args()
    if args.adapter_names is None:
        args.adapter_names = [f"expert_{idx}" for idx in range(len(args.adapters))]
    if len(args.adapter_names) != len(args.adapters):
        raise ValueError("--adapter-names and --adapters must have the same length")
    if args.weights is None:
        args.weights = [1.0] * len(args.adapters)
    if len(args.weights) != len(args.adapters):
        raise ValueError("--weights and --adapters must have the same length")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapters[0], adapter_name=args.adapter_names[0])
    for adapter_path, adapter_name in zip(args.adapters[1:], args.adapter_names[1:], strict=True):
        model.load_adapter(adapter_path, adapter_name=adapter_name)

    merged_adapter_name = f"merged_{args.combination}"
    merge_kwargs: dict[str, Any] = {
        "adapters": args.adapter_names,
        "weights": args.weights,
        "adapter_name": merged_adapter_name,
        "combination_type": args.combination,
    }
    if args.combination in {"ties", "dare_ties"}:
        merge_kwargs["density"] = args.density
        merge_kwargs["majority_sign_method"] = args.majority_sign_method
    call_add_weighted_adapter(model, merge_kwargs)
    model.set_adapter(merged_adapter_name)
    merged = model.merge_and_unload()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(output)
    manifest = {
        "method": args.combination,
        "base_model": args.base_model,
        "adapters": args.adapters,
        "adapter_names": args.adapter_names,
        "weights": args.weights,
        "density": args.density,
        "majority_sign_method": args.majority_sign_method,
        "seed": args.seed,
    }
    with open(output / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved {args.combination} full model to {output}")


if __name__ == "__main__":
    main()
