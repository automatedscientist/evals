# Quantum Harmonic Oscillator Energy Verification Environment

A [Verifiers](https://github.com/willccbb/verifiers) environment for testing language models on quantum harmonic oscillator energy calculations.

## Overview

This environment tests whether language models can compute the expected energy for quantum harmonic oscillator wavefunctions. Models receive a wavefunction as a linear combination of energy eigenstates and must return the exact symbolic energy value.

## Problem Format

Given a 1D quantum harmonic oscillator with ℏ=ω=m=1:
- Potential: V(x) = x²/2
- Energy levels: E_n = n + 1/2
- Wavefunction: ψ(x) = exp(-x²/2) × Σ c_i × H_n(x)

The expected energy is: E = Σ |c_i|² × E_n / Σ |c_i|²

## Installation

```bash
# Clone and setup
git clone <repository>
cd harmonic_osc

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# For Ollama testing
ollama serve
ollama pull qwen2.5:0.5b-instruct
```

## Usage

### Quick Test with Ollama
```bash
.venv/bin/vf-eval vf_quantum_energy \
    --model qwen2.5:0.5b-instruct \
    --api-base-url http://localhost:11434/v1 \
    --num-examples 5 \
    --rollouts-per-example 3 \
    --verbose
```

### Test with OpenRouter (GPT-4o-mini)
```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"

# Run evaluation
.venv/bin/vf-eval vf_quantum_energy \
    --model gpt-4o-mini \
    --api-base-url https://openrouter.ai/api/v1 \
    --api-key-var OPENROUTER_API_KEY \
    --num-examples 50 \
    --rollouts-per-example 3 \
    --verbose
```

## Example Problems

### Example 1
**Input JSON:**
```json
{"input": {"potential": "x**2/2", "wavefunction": "(-128*x**5 + 640*x**3 - 486*x - 4)*exp(-x**2/2)"}}
```
**Expected Output:**
```json
{"expected_energy": "207/74"}
```

### Example 2
**Input JSON:**
```json
{"input": {"potential": "x**2/2", "wavefunction": "(32*x**3 - 56*x - 2)*exp(-x**2/2)"}}
```
**Expected Output:**
```json
{"expected_energy": "97/66"}
```

### Example 3
**Input JSON:**
```json
{"input": {"potential": "x**2/2", "wavefunction": "(-32*x**4 + 96*x**2 - 24)*exp(-x**2/2)"}}
```
**Expected Output:**
```json
{"expected_energy": "9/2"}
```

## Test Results

### Results Summary

| Model | Parameters | Accuracy | Successful Solves |
|-------|------------|----------|-------------------|
| qwen2.5:0.5b-instruct (Ollama) | 0.5B | 0% | 0/15 |
| gpt-4o-mini (OpenRouter) | ~8B | 2% | 3/150 |

### GPT-4o-mini Detailed Results
```
Environment: vf_quantum_energy
Model: gpt-4o-mini
Provider: https://openrouter.ai/api/v1
Examples: 50
Rollouts per example: 3

--- Results ---
reward: avg - 0.020, std - 0.140
Successful solves:
- Problem 27: Solved 2/3 rollouts
- Problem 38: Solved 1/3 rollouts

Example output:
Input: "(-16*x**4 + 16*x**3 + 48*x**2 - 24*x - 11)*exp(-x**2/2)"
Model response: {"expected_energy": "55/3"}
```

## Why Small Models Struggle

This task requires:
1. Understanding quantum mechanics
2. Symbolic math manipulation
3. Exact expressions (not numerical approximations)
4. Proper JSON formatting

Models under 7B parameters typically achieve near 0% accuracy.

## Files

```
harmonic_osc/
├── vf_quantum_energy.py   # Main environment
├── pyproject.toml          # Package configuration
├── README.md              # This file
└── .gitignore            # Git ignore file
```

