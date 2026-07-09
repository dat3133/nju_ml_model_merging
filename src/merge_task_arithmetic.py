from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA experts with full-model task arithmetic.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapters", nargs="+", required=True, help="PEFT adapter paths.")
    parser.add_argument("--lambdas", nargs="+", type=float, required=True, help="Task-vector coefficients.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf-cache")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
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


def load_base(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=False,
    )


def load_merged_expert(args: argparse.Namespace, adapter_path: str):
    from peft import PeftModel

    model = load_base(args)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model.merge_and_unload()


def main() -> None:
    args = parse_args()
    if len(args.adapters) != len(args.lambdas):
        raise ValueError("--adapters and --lambdas must have the same length")

    import gc
    import torch
    from transformers import AutoTokenizer

    merged_model = load_base(args)
    merged_state = merged_model.state_dict()
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in merged_state.items()}

    for adapter_path, coeff in zip(args.adapters, args.lambdas, strict=True):
        print(f"loading expert adapter={adapter_path} lambda={coeff}", flush=True)
        expert = load_merged_expert(args, adapter_path)
        expert_state = expert.state_dict()
        for name, base_tensor in base_state.items():
            if name not in expert_state:
                continue
            if not torch.is_floating_point(base_tensor):
                continue
            target = merged_state[name]
            delta = expert_state[name].detach().cpu().float() - base_tensor.float()
            updated = target.detach().cpu().float() + float(coeff) * delta
            target.copy_(updated.to(dtype=target.dtype))
        del expert, expert_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(output)
    manifest: dict[str, Any] = {
        "method": "task_arithmetic",
        "base_model": args.base_model,
        "adapters": args.adapters,
        "lambdas": args.lambdas,
        "dtype": args.dtype,
    }
    with open(output / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved task-arithmetic full model to {output}")


if __name__ == "__main__":
    main()
