# LLM Weight Distribution Analyzer

## Overview
A Python tool for analyzing and visualizing weight distributions in Large Language Models (LLMs), specifically designed for Llama and Qwen architectures. This tool provides insights into layer-wise weight distributions, statistical patterns, and sparsity analysis.

## Features
- Weight distribution analysis across model layers
- Statistical metrics calculation (mean, std dev, min/max values)
- Sparsity analysis for each layer
- Visualization of weight distributions using violin plots
- Layer-wise statistics comparison
- Support for both Llama and Qwen architectures

## Requirements
```python
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
