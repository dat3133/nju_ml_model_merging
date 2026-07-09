from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Any

from common import format_answer, prompt_for_training, read_jsonl, select_limit


DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA expert on prepared MCQ JSONL data.")
    parser.add_argument("--config", default=None, help="Optional YAML config. CLI values override it.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    return merge_config(parser.parse_args())


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "train_file": "data/processed/medmcqa_train.jsonl",
        "eval_file": None,
        "output_dir": "adapters/lora",
        "cache_dir": "/root/autodl-tmp/hf-cache",
        "task_name": "mcq",
        "max_seq_length": 1024,
        "max_train_samples": 0,
        "max_eval_samples": 0,
        "max_steps": -1,
        "resume_from_checkpoint": None,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": DEFAULT_TARGET_MODULES,
        "learning_rate": 2.0e-4,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 200,
        "save_total_limit": 2,
        "seed": 42,
        "bf16": True,
        "fp16": False,
        "qlora": False,
        "gradient_checkpointing": True,
        "use_chat_template": True,
        "trust_remote_code": True,
    }
    config_values: dict[str, Any] = {}
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            config_values = yaml.safe_load(f) or {}
    merged = dict(defaults)
    merged.update(config_values)
    for key, value in vars(args).items():
        if key == "config":
            merged[key] = value
        elif value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def torch_dtype(args: argparse.Namespace):
    import torch

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def render_training_parts(tokenizer: Any, record: dict[str, Any], *, use_chat_template: bool) -> tuple[str, str]:
    prompt = prompt_for_training(record)
    answer = format_answer(record, leading_space=True)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt, answer


def tokenize_record(tokenizer: Any, record: dict[str, Any], *, max_seq_length: int, use_chat_template: bool) -> dict[str, Any]:
    prompt, answer = render_training_parts(tokenizer, record, use_chat_template=use_chat_template)
    if tokenizer.eos_token:
        answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_seq_length:
        raise ValueError("max_seq_length is too small to fit the answer tokens")
    overflow = len(prompt_ids) + len(answer_ids) - max_seq_length
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class TokenizedDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int, use_chat_template: bool):
        self.items = [
            tokenize_record(
                tokenizer,
                row,
                max_seq_length=max_seq_length,
                use_chat_template=use_chat_template,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        input_ids = torch.full((len(features), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for idx, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[idx, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[idx, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            labels[idx, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_training_arguments(args: argparse.Namespace):
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
    }
    if args.eval_file:
        kwargs["eval_steps"] = args.eval_steps
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            kwargs["eval_strategy"] = "steps"
        else:
            kwargs["evaluation_strategy"] = "steps"
    else:
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            kwargs["eval_strategy"] = "no"
        else:
            kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype(args),
        "device_map": "auto",
    }
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype(args),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        target_modules = [module.strip() for module in str(args.target_modules).split(",") if module.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train_file)
    train_rows = select_limit(train_rows, args.max_train_samples, args.seed)
    eval_dataset = None
    if args.eval_file:
        eval_rows = read_jsonl(args.eval_file)
        eval_rows = select_limit(eval_rows, args.max_eval_samples, args.seed + 1)
        eval_dataset = TokenizedDataset(eval_rows, tokenizer, args.max_seq_length, args.use_chat_template)
    train_dataset = TokenizedDataset(train_rows, tokenizer, args.max_seq_length, args.use_chat_template)
    collator = DataCollatorForCausalLM(tokenizer)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
