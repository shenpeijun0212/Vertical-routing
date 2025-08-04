# Vertical Routing: A Cost-Efficient Collaboration Routing Framework

This repository contains the resources and code used for evaluating large language models across diverse tasks using templated prompting strategies. It is structured to support easy experimentation, evaluation, and extension.

### `data`

This folder contains the benchmark datasets used in our experiments:

- **GSM8K**: A dataset for grade school math word problems.
- **MATH**: A high school-level mathematical reasoning benchmark.
- **MBPP**: The Mostly Basic Python Programming dataset for code generation tasks.
- **WMT**: A machine translation dataset from the WMT benchmark.

Each dataset is formatted for compatibility with the evaluation scripts and prompt templates provided in this repository.

### `eval`

This directory includes the main evaluation scripts used to benchmark model performance on the tasks. These scripts are modular and support both default and domain-specific prompting strategies.

### `templates`

This folder contains the prompt templates used to guide model behavior:

- **Default Template**: A general-purpose template designed to work across all tasks without task-specific customization.
- **Domain-Specific Template**: Customized templates tailored to specific domains such as math, code generation, and translation, aiming to improve task-specific performance.

### `prompt`

Contains the actual prompt examples used during experimentation. These are used in conjunction with the templates to construct full input sequences for the models.

### `result`

This folder includes partial results from selected experiments. The results are organized by task and template type, and include metrics such as accuracy, variance, and token usage.

## Getting Started

To reproduce the experiments:

1. Place the required datasets under the `data/` directory.
2. Choose the appropriate evaluation script in the `eval/` folder.
3. Select a template from the `templates/` folder and a corresponding prompt from `prompt/`.
4. Run the script and inspect the outputs in `result/`.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or collaborations, please open an issue or reach out via email.

