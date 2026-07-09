# Medical-Legal LoRA Model Merging

本仓库是“模型合并与知识复用”课程项目的提交版本。项目以 `Qwen/Qwen2.5-1.5B-Instruct` 为统一基座，分别训练医学和法律两个 LoRA 专家，并比较 Task Arithmetic、TIES、DARE-TIES 和 KnOTS-TIES 风格方法在双任务能力保留上的效果。

完整实验报告见：

- `MODEL MERGING WITH SVD TO TIE THE KNOTS.pdf`

## 项目内容

### Level 1

Level 1 研究两个领域专家模型的合并：

- Medical Expert：在 MedMCQA 医学选择题上训练 LoRA。
- Legal Expert：在 CaseHOLD 法律 holding 选择题上训练 LoRA。
- 统一评测：对候选答案 `" A"` 到 `" E"` 计算条件 log-prob，选择分数最高的选项。
- 合并方法：Task Arithmetic、TIES、DARE-TIES、KnOTS-TIES 风格 SVD 合并迁移版。

主要结论是：两个专家都能提升本领域任务，但会损伤另一个领域；在本项目设置下，Task Arithmetic 0.7/0.7 是最好的双任务折中。

### Level 2

Level 2 复现 ICLR 2025 论文 *Model Merging with SVD to Tie the KnOTS* 的官方代码。

- 官方代码目录：`external/knots`
- 论文 PDF：`MODEL MERGING WITH SVD TO TIE THE KNOTS.pdf`
- 复现设置：ViT-B/32 rank-16 KnOTS-TIES，使用官方发布的 8 个视觉任务 LoRA adapters，只评估 MNIST test。
- 本地复现结果：MNIST normalized accuracy `68.9829`。
- 论文 Table 1 参考值：MNIST normalized accuracy `68.9`。

## 提交文件说明

本提交保留以下内容：

- 实验报告：`MODEL MERGING WITH SVD TO TIE THE KNOTS.pdf`
- 源代码：`src/`
- 运行脚本：`scripts/`
- 配置文件：`configs/`
- 环境文件：`environment.yml`、`requirements.txt`
- 结果表：`results/tables/`
- 结果图：`results/figures/`
- 评测 summary：`results/raw_merged/*.summary.json`
- Level 2 最小复现代码：`external/knots/`

以下大文件没有随提交保留：

- `adapters/`：训练得到的 LoRA adapter 权重。
- `merged/`：合并后的 full model 权重。
- 逐样本评测 CSV。
- 外部 KnOTS 运行时下载的数据文件和生成的 CLIP head。
- Hugging Face 模型缓存。

这些文件体积较大，可以通过下面的命令重新生成。

## 目录结构

```text
.
├── configs/                 # 训练与实验配置
├── data/                    # 已处理数据与 split id
├── external/knots/          # KnOTS 官方代码及本项目新增的 MNIST 复现入口
├── results/
│   ├── figures/             # 报告使用的图
│   ├── raw_merged/          # 评测 summary JSON
│   └── tables/              # 报告使用的结果表
├── scripts/                 # 一键运行脚本
├── src/                     # 数据处理、训练、合并、评测和绘图代码
├── MODEL MERGING WITH SVD TO TIE THE KNOTS.pdf
└── README.md
```

## 环境配置

建议使用 CUDA 可用的 Linux 环境。项目实验使用过的基座模型和主要依赖如下：

- Base model：`Qwen/Qwen2.5-1.5B-Instruct`
- Python：3.10
- PyTorch：CUDA 版本
- Transformers / PEFT / Datasets / Pandas / Matplotlib / Seaborn

创建环境：

```bash
conda env create -f environment.yml
conda activate merge_lora
```

或在已有环境中安装依赖：

```bash
pip install -r requirements.txt
```

常用环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
```

如需记录当前环境：

```bash
bash scripts/collect_env.sh
```

该命令会生成 `environment_report.md`。

## Level 1 复现流程

### 1. 准备数据

```bash
python src/prepare_medmcqa.py --cache-dir /root/autodl-tmp/hf-cache
python src/prepare_casehold.py --cache-dir /root/autodl-tmp/hf-cache
```

处理后的主文件包括：

- `data/processed/medmcqa_train.jsonl`
- `data/processed/medmcqa_val.jsonl`
- `data/processed/medmcqa_test.jsonl`
- `data/processed/casehold_train.jsonl`
- `data/processed/casehold_val.jsonl`
- `data/processed/casehold_test.jsonl`

注意：本项目的 `medmcqa_test.jsonl` 是从 MedMCQA 官方 validation split 中拆出的 held-out test，不是官方隐藏 test。

### 2. 训练两个 LoRA 专家

```bash
python src/train_lora.py \
  --config configs/train_medmcqa.yaml \
  --output-dir adapters/medical_lora_full

python src/train_lora.py \
  --config configs/train_casehold.yaml \
  --output-dir adapters/legal_lora_full
```

### 3. 运行 Level 1 合并 sweep

```bash
MED_ADAPTER=adapters/medical_lora_full \
LEGAL_ADAPTER=adapters/legal_lora_full \
MERGED_DIR=merged \
MODEL_PREFIX=full_ \
bash scripts/run_level1_sweep.sh
```

该脚本会生成 Task Arithmetic、TIES 和 DARE-TIES 合并模型。KnOTS-TIES 迁移版需要单独运行：

```bash
python src/merge_knots.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapters adapters/medical_lora_full adapters/legal_lora_full \
  --density 0.4 \
  --svd-rank 16 \
  --device cuda \
  --output-dir merged/full_knots_ties_d04_r16 \
  --cache-dir /root/autodl-tmp/hf-cache
```

### 4. 评测并生成结果表和图片

完整评测命令：

```bash
MED_ADAPTER=adapters/medical_lora_full \
LEGAL_ADAPTER=adapters/legal_lora_full \
RESULTS_DIR=results/raw_merged \
TABLE_OUTPUT=results/tables/full_main_results.csv \
FIGURE_DIR=results/figures \
MERGED_MODEL_LIST=merged/full_level1_sweep_models.txt \
bash scripts/evaluate_all_models.sh
```

如果需要把 KnOTS-TIES 迁移版加入同一张结果表，需要额外评测 `merged/full_knots_ties_d04_r16`，再运行：

```bash
python src/plot_results.py \
  --summaries 'results/raw_merged/*.summary.json' \
  --table-output results/tables/full_main_results.csv \
  --figure-dir results/figures
```

本提交已经保留最终生成的表和图。

## Level 2 复现流程

Level 2 使用单独的 KnOTS 环境。进入官方代码目录后安装依赖：

```bash
cd external/knots
PIP_NO_CACHE_DIR=1 conda env update --file conda_environment.autodl.yaml --prune
conda activate knots_env
```

运行 MNIST 单项复现：

```bash
cd /root/autodl-tmp/pro
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache

bash scripts/run_level2_knots_mnist.sh
```

输出表：

- `results/tables/level2_knots_mnist.csv`

## 已提交结果文件

主要表格：

- `results/tables/full_main_results.csv`
- `results/tables/dare_ties_mean_std.csv`
- `results/tables/level2_knots_mnist.csv`

主要图：

- `results/figures/dual_task_bar.png`
- `results/figures/task_arithmetic_heatmap.png`
- `results/figures/ties_density_curve.png`
- `results/figures/dual_task_pareto_scatter.png`

评测 summary：

- `results/raw_merged/*.summary.json`

## 关键结果

Level 1 最佳合并模型：

| Method | MedMCQA test | CaseHOLD test | Test avg |
| --- | ---: | ---: | ---: |
| Task Arithmetic 0.7/0.7 | 0.5368 | 0.8186 | 0.6777 |

Level 2 官方复现：

| Method | Dataset | Accuracy | Normalized Acc | Paper Ref. |
| --- | --- | ---: | ---: | ---: |
| KnOTS-TIES 8-merge | MNIST test | 68.5000 | 68.9829 | 68.9 |

完整分析见 `MODEL MERGING WITH SVD TO TIE THE KNOTS.pdf`。
