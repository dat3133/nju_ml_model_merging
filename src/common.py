from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

LETTERS = "ABCDE"

TASK_TITLES = {
    "medmcqa": "medical multiple-choice exam",
    "casehold": "legal case holding selection",
}

TASK_INSTRUCTIONS = {
    "medmcqa": "Choose the single best answer to the medical multiple-choice question.",
    "casehold": "Choose the holding that best completes the legal reasoning.",
}


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_ids(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["id"]) + "\n")
            count += 1
    return count


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def stable_id(prefix: str, source_split: str, idx: int, payload: dict[str, Any]) -> str:
    key = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}-{source_split}-{idx}-{stable_hash(key)}"


def select_limit(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [rows[i] for i in keep]


def deterministic_split(
    rows: list[dict[str, Any]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    cut = max(1, min(len(rows) - 1, math.floor(len(rows) * (1.0 - test_ratio))))
    return rows[:cut], rows[cut:]


def parse_label(raw_label: Any, choices: list[str], *, label_base: str = "zero") -> int | None:
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return None
    if isinstance(raw_label, int):
        if label_base == "one" and 1 <= raw_label <= len(choices):
            return raw_label - 1
        if 0 <= raw_label < len(choices):
            return raw_label
        if label_base == "auto" and 1 <= raw_label <= len(choices):
            return raw_label - 1
        return None
    text = str(raw_label).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in LETTERS[: len(choices)]:
        return LETTERS.index(upper)
    if text.isdigit():
        return parse_label(int(text), choices, label_base=label_base)
    normalized = " ".join(text.lower().split())
    for idx, choice in enumerate(choices):
        if normalized == " ".join(str(choice).lower().split()):
            return idx
    return None


def clean_choice(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\n", " ").split())


def is_labeled_record(row: dict[str, Any]) -> bool:
    label = row.get("label")
    choices = row.get("choices") or []
    return isinstance(label, int) and 0 <= label < len(choices)


def answer_letter(label: int) -> str:
    return LETTERS[label]


def build_prompt(record: dict[str, Any], *, include_task_instruction: bool = True) -> str:
    task = record.get("task", "mcq")
    instruction = TASK_INSTRUCTIONS.get(task, "Choose the single best answer.")
    question = record.get("question") or record.get("context") or record.get("prompt")
    if question is None:
        raise KeyError("record must contain question, context, or prompt")

    parts: list[str] = []
    if include_task_instruction:
        parts.append(instruction)
        parts.append("")
    if task == "casehold":
        parts.append("Context:")
    else:
        parts.append("Question:")
    parts.append(str(question).strip())
    parts.append("")
    parts.append("Options:")
    for idx, choice in enumerate(record["choices"]):
        parts.append(f"{LETTERS[idx]}. {choice}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def prompt_for_training(record: dict[str, Any]) -> str:
    return build_prompt(record).rstrip()


def format_answer(record: dict[str, Any], *, leading_space: bool = True) -> str:
    answer = answer_letter(int(record["label"]))
    return f" {answer}" if leading_space else answer


def make_robustness_records(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    limit: int | None,
    split_name: str = "robustness",
) -> list[dict[str, Any]]:
    base_rows = select_limit(rows, limit, seed)
    robust_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(base_rows):
        choices = list(row["choices"])
        if len(choices) < 2:
            continue
        rng = random.Random(f"{seed}:{row['id']}")
        order = list(range(len(choices)))
        rng.shuffle(order)
        if order == list(range(len(choices))):
            order = order[1:] + order[:1]
        old_label = int(row["label"])
        new_label = order.index(old_label)
        new_row = dict(row)
        new_row["id"] = f"{row['id']}-robust-{idx}"
        new_row["source_id"] = row["id"]
        new_row["split"] = split_name
        new_row["choices"] = [choices[i] for i in order]
        new_row["label"] = new_label
        metadata = dict(new_row.get("metadata") or {})
        metadata["robustness"] = "choice_shuffle"
        metadata["choice_order"] = order
        new_row["metadata"] = metadata
        robust_rows.append(new_row)
    return robust_rows


def summarize_split(name: str, rows: list[dict[str, Any]]) -> str:
    labels = {letter: 0 for letter in LETTERS}
    for row in rows:
        label = row.get("label")
        if isinstance(label, int) and 0 <= label < len(LETTERS):
            labels[LETTERS[label]] += 1
    nonzero = ", ".join(f"{k}:{v}" for k, v in labels.items() if v)
    return f"{name}: {len(rows)} rows ({nonzero or 'no labels'})"
