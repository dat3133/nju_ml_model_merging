from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from common import LETTERS, build_prompt, iter_jsonl, read_jsonl, select_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on prepared MCQ JSONL files.")
    parser.add_argument("--model", required=True, help="Base/full model path or HF model id.")
    parser.add_argument("--base-model", default=None, help="Base model id/path when --adapter is used.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--input", required=True, help="Prepared JSONL file.")
    parser.add_argument("--output", required=True, help="Per-example CSV output.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON output.")
    parser.add_argument("--audit-file", default=None, help="Optional JSONL sample dump for manual inspection.")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf-cache")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use empty string to disable.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-by-length", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return "auto"
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(dtype_name)


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.base_model if args.adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_source = args.base_model if args.adapter else args.model
    model_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    if not args.device_map and torch.cuda.is_available():
        model.cuda()
    return model, tokenizer


def render_prompt(tokenizer: Any, record: dict[str, Any], *, use_chat_template: bool) -> str:
    prompt = build_prompt(record).rstrip()
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def score_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    use_chat_template: bool,
    normalize_by_length: bool,
) -> list[dict[str, Any]]:
    import torch

    sequences: list[list[int]] = []
    spans: list[tuple[int, int, int, int]] = []
    rendered_prompts: list[str] = []
    for example_idx, record in enumerate(records):
        prompt_text = render_prompt(tokenizer, record, use_chat_template=use_chat_template)
        rendered_prompts.append(prompt_text)
        prompt_ids = encode_text(tokenizer, prompt_text)
        for choice_idx in range(len(record["choices"])):
            answer_ids = encode_text(tokenizer, f" {LETTERS[choice_idx]}")
            if not answer_ids:
                raise ValueError(f"empty answer tokenization for {LETTERS[choice_idx]}")
            input_ids = prompt_ids + answer_ids
            sequences.append(input_ids)
            spans.append((example_idx, choice_idx, len(prompt_ids), len(input_ids)))

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, : len(seq)] = 1

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    scores = [[-math.inf for _ in record["choices"]] for record in records]
    for flat_idx, (example_idx, choice_idx, start, end) in enumerate(spans):
        token_logprobs: list[float] = []
        for pos in range(start, end):
            if pos == 0:
                continue
            token_id = int(input_ids[flat_idx, pos].item())
            token_logits = logits[flat_idx, pos - 1].float()
            token_logprob = token_logits[token_id] - torch.logsumexp(token_logits, dim=-1)
            token_logprobs.append(float(token_logprob.item()))
        score = sum(token_logprobs)
        if normalize_by_length and token_logprobs:
            score /= len(token_logprobs)
        scores[example_idx][choice_idx] = score

    results: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        pred = max(range(len(scores[idx])), key=lambda choice_idx: scores[idx][choice_idx])
        label = int(record["label"])
        row: dict[str, Any] = {
            "id": record["id"],
            "task": record.get("task", ""),
            "split": record.get("split", ""),
            "label": LETTERS[label],
            "label_index": label,
            "pred": LETTERS[pred],
            "pred_index": pred,
            "correct": int(pred == label),
            "num_choices": len(record["choices"]),
            "prompt": rendered_prompts[idx],
        }
        for choice_idx, choice in enumerate(record["choices"]):
            row[f"choice_{LETTERS[choice_idx]}"] = choice
            row[f"score_{LETTERS[choice_idx]}"] = scores[idx][choice_idx]
        results.append(row)
    return results


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "id",
        "task",
        "split",
        "label",
        "label_index",
        "pred",
        "pred_index",
        "correct",
        "num_choices",
    ]
    score_fields = [f"score_{letter}" for letter in LETTERS]
    choice_fields = [f"choice_{letter}" for letter in LETTERS]
    fields = base_fields + score_fields + choice_fields + ["prompt"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_audit(path: str, rows: list[dict[str, Any]], limit: int = 20) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows[:limit]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    records = read_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        records = select_limit(records, args.max_samples, args.seed)
    if not records:
        raise ValueError(f"no records loaded from {args.input}")

    model, tokenizer = load_model_and_tokenizer(args)
    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        all_rows.extend(
            score_batch(
                model,
                tokenizer,
                batch,
                use_chat_template=args.use_chat_template,
                normalize_by_length=args.normalize_by_length,
            )
        )
        print(f"evaluated {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)

    write_csv(args.output, all_rows)
    accuracy = sum(row["correct"] for row in all_rows) / len(all_rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "input": args.input,
        "output": args.output,
        "n": len(all_rows),
        "accuracy": accuracy,
        "normalize_by_length": args.normalize_by_length,
        "use_chat_template": args.use_chat_template,
    }
    summary_output = args.summary_output or str(Path(args.output).with_suffix(".summary.json"))
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.audit_file:
        write_audit(args.audit_file, all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
