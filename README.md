# scope-decoding

This repository contains code for launching SCOPE unfaithful decoding methods.

$y_t \sim p_{\theta}(\cdot | y_{<t}, x) + \alpha_t p_{\text{LM}}(\cdot | y_{<t}),$
where $\alpha_t \in \{0, 1\}$ is a binary random variable that indicates whether the LM is used at time $t$.

# Quick start

## Installation
First, install your PyTorch version (>= 2.0 should work).
```bash
pip install -e .
```

## Launching generation

```bash
python src/scope/example.py main_model_path="YOUR MODEL" noise_model_path="YOUR MODEL" mixture_alpha=0.3
```

# Distributed generation

## Data preparation
Make sure you prepare a huggingface dataset with the following keys: "main_input_ids", "noise_input_ids".
The "main_input_ids" should be the input ids of the main model, and the "noise_input_ids" should be the input ids of the non-contextualized model (for example empty string, or just a general style conditioning sequence).

This code is designed to be used on a SLURM cluster, but should work on other parallelization systems with some modifications.
For single task generation, it should work out of the box:

```bash
python src/scope/launcher.py main_model.model_path="YOUR MODEL" noise_model.model_path="YOUR MODEL" generation.mixture_alpha=0.3 data.dataset_path="YOUR DATA" out_path=output
```