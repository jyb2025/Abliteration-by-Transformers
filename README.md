# Abliteration - All-in-one Script

A comprehensive tool for removing refusal directions from Large Language Models (LLMs), making them more helpful and less likely to refuse valid requests.

## Overview

This script combines all the functionality from the [Abliteration](https://github.com/Orion-zhen/abliteration.git) codebase into a single, easy-to-use file. It allows you to:

1. **Calculate refusal directions** - Identify the most significant refusal directions in a model using harmful and harmless prompts
2. **Remove refusal directions** - Abliterate (remove) these directions from the model weights
3. **Chat with the model** - Test the original or abliterated model through an interactive chat interface
4. **Compare models** - Analyze differences between original and abliterated models

## How It Works

Abliteration works by:

1. Identifying the "refusal direction" in the model's weight space by comparing model activations on harmful vs. harmless prompts
2. Removing this direction from key weight matrices in the model's transformer layers
3. Preserving the model's general capabilities while reducing its tendency to refuse valid requests

## Installation

### Requirements

- Python 3.9+
- PyTorch
- Transformers
- Datasets
- tqdm
- pandas
- psutil

Install dependencies:

```bash
pip install transformers datasets torch tqdm pandas psutil
```

## Usage

### Abliterate a Model

```bash
python abliterate_all_in_one.py abliterate -m <model_path> -o <output_dir>
```

Example:
```bash
python abliterate_all_in_one.py abliterate -m meta-llama/Llama-3.2-3B-Instruct -o llama3.2-3b-abliterated
```

### Chat with a Model

```bash
python abliterate_all_in_one.py chat -m <model_path>
```

Example:
```bash
python abliterate_all_in_one.py chat -m llama3.2-3b-abliterated
```

### Compare Original and Abliterated Models

```bash
python abliterate_all_in_one.py compare --original <original_model_path> --abliterated <abliterated_model_path>
```

Example:
```bash
python abliterate_all_in_one.py compare --original meta-llama/Llama-3.2-3B-Instruct --abliterated llama3.2-3b-abliterated --output-file comparison_report.md
```

## Advanced Options

### Abliteration Options

- `--skip-begin`: Number of layers to skip at the beginning (default: 1)
- `--skip-end`: Number of layers to skip at the end (default: 0)
- `--scale-factor`: Scale factor for ablation (default: 1.0)
- `--top-refusal-layers`: Only abliterate the N layers with highest refusal factors
- `--specific-layers`: Comma-separated list of specific layer indices to abliterate
- `--proportional-scaling`: Scale abliteration proportionally to each layer's refusal factor
- `--force-abliteration`: Force abliteration even when refusal factors are negligible

### Device Options

- `--device`: Target device (auto, cuda, cpu, last-gpu)
- `--gpu-id`: Specific GPU ID to use
- `--multi-gpu`: Distribute model across multiple GPUs
- `--max-memory`: Maximum memory to use per GPU

### Precision Options

- `--precision`: Precision to use (fp16, bf16, fp32)
- `--load-in-4bit`: Load model in 4-bit precision
- `--load-in-8bit`: Load model in 8-bit precision
- `--flash-attn`: Use flash attention 2

### Data Options

- `--data-harmful`: Custom harmful prompts file
- `--data-harmless`: Custom harmless prompts file
- `--num-harmful`: Number of harmful calibrations to randomly select
- `--num-harmless`: Number of harmless calibrations to randomly select

## Technical Details

The script performs the following key operations:

1. **Refusal Direction Calculation**:
   - Computes representations for harmful and harmless prompts
   - Calculates the difference vector (refusal direction)
   - Normalizes this direction for consistent application

2. **Layer Analysis**:
   - Analyzes each layer to determine its contribution to refusal behavior
   - Ranks layers by their refusal factors
   - Allows targeting specific layers for abliteration

3. **Tensor Modification**:
   - Applies a projection-based modification to weight matrices
   - Removes components aligned with the refusal direction
   - Preserves other capabilities of the model

4. **Model Comparison**:
   - Provides detailed analysis of changes between original and abliterated models
   - Reports on parameter changes at layer and component levels
   - Generates comprehensive comparison reports

## Acknowledgments

This script is based on the [Abliteration](https://github.com/Orion-zhen/abliteration.git) project, which pioneered the technique of removing refusal directions from LLMs. The all-in-one script consolidates the functionality from the original repository into a single, easy-to-use file.

## License

This project is released under the same license as the original Abliteration repository.
