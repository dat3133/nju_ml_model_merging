from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MCQ evaluation summaries and generate project figures.")
    parser.add_argument("--summaries", nargs="+", default=["results/raw/*.summary.json"])
    parser.add_argument("--table-output", default="results/tables/main_results.csv")
    parser.add_argument("--figure-dir", default="results/figures")
    return parser.parse_args()


def expand_patterns(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    return sorted(set(paths))


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_task_split(input_path: str) -> tuple[str, str]:
    stem = Path(input_path).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    return stem, ""


def load_manifest(model_path: str | None) -> dict[str, Any]:
    if not model_path:
        return {}
    manifest_path = Path(model_path) / "merge_manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    return {}


def infer_method(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    if manifest.get("method"):
        method = str(manifest["method"])
        if method == "task_arithmetic":
            lambdas = manifest.get("lambdas") or []
            return f"Task Arithmetic {lambdas}"
        if method in {"ties", "dare_ties"}:
            return f"{method.upper()} d={manifest.get('density')}"
        if method.startswith("knots"):
            return f"KnOTS-TIES d={manifest.get('density')}"
        return method
    adapter = summary.get("adapter")
    if adapter:
        adapter_text = str(adapter).lower()
        if "medical" in adapter_text or "med" in adapter_text:
            return "Medical Expert"
        if "legal" in adapter_text or "case" in adapter_text:
            return "Legal Expert"
        return f"Adapter {Path(str(adapter)).name}"
    return "Base"


def rows_from_summaries(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        summary = read_json(path)
        manifest = load_manifest(summary.get("model"))
        task, split = infer_task_split(str(summary.get("input", "")))
        row = {
            "summary_path": path,
            "method": infer_method(summary, manifest),
            "raw_method": manifest.get("method", "adapter" if summary.get("adapter") else "base"),
            "task": task,
            "split": split,
            "n": summary.get("n"),
            "accuracy": summary.get("accuracy"),
            "model": summary.get("model"),
            "adapter": summary.get("adapter"),
            "input": summary.get("input"),
            "density": manifest.get("density"),
            "seed": manifest.get("seed"),
            "lambda_med": None,
            "lambda_legal": None,
        }
        lambdas = manifest.get("lambdas")
        if isinstance(lambdas, list) and len(lambdas) >= 2:
            row["lambda_med"] = lambdas[0]
            row["lambda_legal"] = lambdas[1]
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, Any]], output: str) -> None:
    import pandas as pd

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"wrote {output} ({len(df)} rows)")


def plot_dual_task_bar(df: Any, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df[df["split"].eq("test")].copy()
    if plot_df.empty:
        print("skip dual-task bar: no test rows")
        return
    plt.figure(figsize=(max(8, 0.7 * plot_df["method"].nunique()), 5))
    sns.barplot(data=plot_df, x="method", y="accuracy", hue="task")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output = figure_dir / "dual_task_bar.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"wrote {output}")


def plot_task_arithmetic_heatmap(df: Any, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df[df["raw_method"].eq("task_arithmetic") & df["split"].eq("val")].copy()
    if plot_df.empty or plot_df["lambda_med"].isna().all():
        print("skip task arithmetic heatmap: no val task-arithmetic rows")
        return
    pivot = plot_df.pivot_table(
        values="accuracy",
        index="lambda_med",
        columns="lambda_legal",
        aggfunc="mean",
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.ylabel("Medical lambda")
    plt.xlabel("Legal lambda")
    plt.tight_layout()
    output = figure_dir / "task_arithmetic_heatmap.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"wrote {output}")


def plot_ties_density_curve(df: Any, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df[df["raw_method"].isin(["ties", "dare_ties", "knots_ties_lora_svd"]) & df["split"].eq("val")].copy()
    plot_df = plot_df[plot_df["density"].notna()]
    if plot_df.empty:
        print("skip TIES density curve: no val density rows")
        return
    plot_df["family"] = plot_df["raw_method"].replace({"knots_ties_lora_svd": "knots_ties"})
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=plot_df, x="density", y="accuracy", hue="task", style="family", marker="o", errorbar="sd")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Density")
    plt.tight_layout()
    output = figure_dir / "ties_density_curve.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"wrote {output}")


def write_dare_mean_std(df: Any, table_dir: Path) -> None:
    dare = df[df["raw_method"].eq("dare_ties")].copy()
    if dare.empty:
        print("skip DARE mean/std table: no DARE-TIES rows")
        return
    grouped = (
        dare.groupby(["density", "task", "split"], dropna=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output = table_dir / "dare_ties_mean_std.csv"
    grouped.to_csv(output, index=False)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    summary_paths = expand_patterns(args.summaries)
    if not summary_paths:
        raise FileNotFoundError("no summary files matched")
    rows = rows_from_summaries(summary_paths)
    write_table(rows, args.table_output)
    df = pd.DataFrame(rows)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(args.table_output).parent
    plot_dual_task_bar(df, figure_dir)
    plot_task_arithmetic_heatmap(df, figure_dir)
    plot_ties_density_curve(df, figure_dir)
    write_dare_mean_std(df, table_dir)


if __name__ == "__main__":
    main()
