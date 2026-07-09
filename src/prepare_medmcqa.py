from __future__ import annotations

import argparse
from typing import Any

from common import (
    clean_choice,
    deterministic_split,
    is_labeled_record,
    make_robustness_records,
    parse_label,
    select_limit,
    stable_id,
    summarize_split,
    write_ids,
    write_jsonl,
)


def build_record(row: dict[str, Any], *, source_split: str, idx: int, label_base: str) -> dict[str, Any] | None:
    choices = [
        clean_choice(row.get("opa")),
        clean_choice(row.get("opb")),
        clean_choice(row.get("opc")),
        clean_choice(row.get("opd")),
    ]
    if row.get("ope") not in (None, ""):
        choices.append(clean_choice(row.get("ope")))
    choices = [choice for choice in choices if choice]
    if len(choices) < 2:
        return None

    raw_label = row.get("cop", row.get("label", row.get("answer")))
    label = parse_label(raw_label, choices, label_base=label_base)
    if label is None:
        return None

    question = clean_choice(row.get("question"))
    if not question:
        return None

    return {
        "id": stable_id("medmcqa", source_split, idx, row),
        "task": "medmcqa",
        "source": "openlifescienceai/medmcqa",
        "source_split": source_split,
        "split": source_split,
        "question": question,
        "choices": choices,
        "label": label,
        "metadata": {
            "subject_name": row.get("subject_name"),
            "topic_name": row.get("topic_name"),
            "choice_type": row.get("choice_type"),
            "raw_label": raw_label,
        },
    }


def convert_split(dataset_split: Any, *, source_split: str, label_base: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset_split):
        record = build_record(dict(row), source_split=source_split, idx=idx, label_base=label_base)
        if record is not None and is_labeled_record(record):
            records.append(record)
    return records


def write_split(
    rows: list[dict[str, Any]],
    *,
    task: str,
    split: str,
    output_dir: str,
    split_dir: str,
) -> None:
    for row in rows:
        row["split"] = split
    jsonl_path = f"{output_dir}/{task}_{split}.jsonl"
    ids_path = f"{split_dir}/{task}_{split}_ids.txt"
    write_jsonl(jsonl_path, rows)
    write_ids(ids_path, rows)
    print(summarize_split(split, rows))
    print(f"  wrote {jsonl_path}")
    print(f"  wrote {ids_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MedMCQA as deterministic MCQ JSONL files.")
    parser.add_argument("--dataset", default="openlifescienceai/medmcqa")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/hf-cache")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-base", choices=["zero", "one", "auto"], default="zero")
    parser.add_argument("--train-limit", type=int, default=0, help="0 means keep all.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means keep all.")
    parser.add_argument("--test-limit", type=int, default=0, help="0 means keep all.")
    parser.add_argument("--robustness-limit", type=int, default=1000)
    parser.add_argument(
        "--fallback-test-ratio",
        type=float,
        default=0.5,
        help="If no labeled test split exists, split validation into val/test with this test ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    train_rows = convert_split(dataset["train"], source_split="train", label_base=args.label_base)

    validation_key = "validation" if "validation" in dataset else "val"
    val_rows = convert_split(dataset[validation_key], source_split=validation_key, label_base=args.label_base)
    test_rows: list[dict[str, Any]] = []
    if "test" in dataset:
        test_rows = convert_split(dataset["test"], source_split="test", label_base=args.label_base)
    if not test_rows and len(val_rows) >= 2:
        val_rows, test_rows = deterministic_split(
            val_rows,
            test_ratio=args.fallback_test_ratio,
            seed=args.seed,
        )
        for row in val_rows:
            row["metadata"]["fallback_from"] = validation_key
        for row in test_rows:
            row["metadata"]["fallback_from"] = validation_key

    train_rows = select_limit(train_rows, args.train_limit, args.seed)
    val_rows = select_limit(val_rows, args.val_limit, args.seed + 1)
    test_rows = select_limit(test_rows, args.test_limit, args.seed + 2)
    robustness_rows = make_robustness_records(
        test_rows or val_rows,
        seed=args.seed,
        limit=args.robustness_limit,
    )

    write_split(train_rows, task="medmcqa", split="train", output_dir=args.output_dir, split_dir=args.split_dir)
    write_split(val_rows, task="medmcqa", split="val", output_dir=args.output_dir, split_dir=args.split_dir)
    write_split(test_rows, task="medmcqa", split="test", output_dir=args.output_dir, split_dir=args.split_dir)
    write_split(
        robustness_rows,
        task="medmcqa",
        split="robustness",
        output_dir=args.output_dir,
        split_dir=args.split_dir,
    )

    if train_rows:
        sample = train_rows[0]
        print("sample:", sample["id"], sample["question"][:160], sample["choices"], sample["label"])


if __name__ == "__main__":
    main()
