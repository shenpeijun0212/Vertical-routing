# Vertical Routing

Code and data for **Vertical Routing: A Cost-Efficient Collaboration Routing Framework**.

Vertical Routing studies a simple question: instead of assigning a whole query to either a large language model or a small language model, can we split one answer into stages and route only the most critical stages to the large model?

This repository contains the prompts, processed benchmark files, template implementations, evaluation examples, and selected result files used for the paper.

## Method Overview

Vertical Routing decomposes generation into ordered stages and allocates model capacity at the stage level.

- **Default template**: the large model generates an initial prefix of the response, and the small model continues from that prefix. This is used when no reliable task-specific structure is available.
- **Domain-specific templates**: the large model handles high-criticality subtasks defined by task structure, and the small model completes the remaining generation.
- **Plan-and-Solve (PS)**: for math reasoning, the large model generates a plan and the small model executes the solution.
- **Deep Reasoning Translation (DRT)**: for translation, the large model extracts key terms or constraints and the small model generates the final sentence.

The goal is to preserve most of the quality of the large model while reducing large-model generation cost and improving routing stability.

## Repository Structure

```text
.
|-- data/
|   |-- GSM8K.jsonl
|   |-- MATH.jsonl
|   |-- MBPP.jsonl
|   `-- WMT.json
|-- eval/
|   |-- evaluate_GSM8K.py
|   |-- evaluate_MBPP.py
|   `-- evaluate_WMT.py
|-- prompt/
|   |-- Base_8-shot.txt
|   |-- MBPP_8-shot.txt
|   |-- MT_8-shot.txt
|   `-- DRT_8-shot_gpt.txt
|-- templates/
|   |-- Default template/
|   `-- templates/Domain-specific template/
|-- result/
`-- README.md
```

## Data

The `data/` directory includes processed benchmark files used by the experiments:

| File | Task | Format | Main fields |
| --- | --- | --- | --- |
| `data/GSM8K.jsonl` | Grade-school math reasoning | JSONL | `question`, `answer` |
| `data/MATH.jsonl` | Competition-style math reasoning | JSONL | `problem`, `solution`, `answer`, `subject`, `level` |
| `data/MBPP.jsonl` | Python code generation | JSONL | `task_id`, `prompt`, `code`, `test_list` |
| `data/WMT.json` | English-Chinese translation | JSON | `en`, `zh` |

The files are already placed in the expected repository layout. If you replace them with the original benchmark releases or another split, keep the same field names or adjust the evaluation scripts accordingly.

## Environment

The real GSM8K evaluation script loads Hugging Face causal language models. A typical environment is:

```bash
conda create -n vertical-routing python=3.10
conda activate vertical-routing
pip install torch transformers accelerate
```

For CUDA-enabled inference, install the PyTorch build that matches your CUDA version. The demonstration scripts for MBPP, WMT, and the standalone templates use mock models and can run without downloading large model checkpoints.

## Quick Start

From the repository root:

```bash
cd <repo-root>
```

Run the lightweight template demos:

```bash
python "templates/Default template/Default template.py"
python "templates/templates/Domain-specific template/DRT.py"
python eval/evaluate_MBPP.py
python eval/evaluate_WMT.py
```

These scripts are designed to show the routing workflow. They use mock model objects, so they are useful for understanding the templates but do not reproduce the full paper results.

## Running Real GSM8K Evaluation

`eval/evaluate_GSM8K.py` is the real model-loading evaluation script. Before running it, edit the `Config` class near the top of the file:

```python
LARGE_MODEL_PATH = "/path/to/large/model"
SMALL_MODEL_PATH = "/path/to/small/model"
FEWSHOT_PROMPT_PATH = "prompt/Base_8-shot.txt"
TEST_DATA_PATH = "data/GSM8K.jsonl"
DATA_SIZE = 500
RESULT_SAVE_BASE_PATH = "./results"
```

Then run:

```bash
python eval/evaluate_GSM8K.py
```

The script performs a two-stage Plan-and-Solve workflow:

1. The large model generates a plan for the math problem.
2. The small model receives the original problem and the plan, then generates the solution.
3. The evaluator extracts the final numeric answer marked by `####`.
4. A JSONL result file is saved under `RESULT_SAVE_BASE_PATH`.

The first line of the output JSONL file is a summary. The following lines contain per-example records, including correctness, latency, and token counts for the large and small models.

## Prompts and Templates

The `prompt/` directory stores few-shot prompts:

- `Base_8-shot.txt`: math reasoning prompt.
- `MBPP_8-shot.txt`: code generation prompt.
- `MT_8-shot.txt`: direct machine translation prompt.
- `DRT_8-shot_gpt.txt`: translation prompt for the DRT-style workflow.

The `templates/` directory contains minimal standalone implementations of the routing templates:

- `templates/Default template/Default template.py`
- `templates/templates/Domain-specific template/DRT.py`
- `templates/templates/Domain-specific template/Plan_and_Solve.py`

These files are intentionally compact and readable. They are meant to clarify the stage-level routing logic rather than provide a full training or serving framework.

## Results

The `result/` directory contains selected output files from experiments and baselines, including:

- single-model MBPP outputs;
- large-small collaboration outputs;
- RouterDC and FrugalGPT baseline outputs;
- per-item pass/fail or correctness records.

Result file names encode the model pair, prompt setting, and task where available, for example:

```text
Qwen2.5-32B_Qwen2.5-3B_8-shot_MBPP.json
Qwen2.5-14B_Qwen2.5-7B_8-shot_MBPP.json
routerdc.jsonl
frugalgpt.jsonl
```

When adding new experiments, we recommend storing raw generations and evaluation summaries separately, and recording the model paths, prompt file, decoding parameters, and dataset split.

## Reproducibility Notes

- Model paths in `eval/evaluate_GSM8K.py` are placeholders from the original experiment environment. Update them before running.
- Exact scores depend on the model checkpoint, decoding parameters, token budget, and dataset split.
- The MBPP and WMT evaluation files in this release are runnable demonstrations with mock models. They document the template workflow but are not full reproduction scripts.
- The processed data files are included for convenience. If you use official benchmark data directly, check that answer extraction and field names still match the scripts.
- The paper reports cost-related metrics using generated token counts and the share of large-model output. Keep both large-model and small-model token counts when extending the experiments.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{shen2026verticalrouting,
  title = {Vertical Routing: A Cost-Efficient Collaboration Routing Framework},
  author = {Shen, Si and Shen, Peijun and Zhu, Danhao},
  journal = {Transactions of the Association for Computational Linguistics},
  year = {2026},
  note = {To appear}
}
```

Please replace this placeholder entry with the official citation once the paper metadata is available.

## Contact

For questions, please open an issue on the repository or contact the authors:

- Si Shen: `shensi@njust.edu.cn`
- Peijun Shen: `124106022972@njust.edu.cn`
- Danhao Zhu: `zhudanhao@jspi.cn`
