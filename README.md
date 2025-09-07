# Theoretical Physics Evaluation Pipeline

This repository contains evaluation environments for testing language models on theoretical physics problems. Each subdirectory represents a specific physics domain with its own evaluation framework.

## Evaluations

### 1. Quantum Harmonic Oscillator (harmonic_osc/)

**Domain:** Quantum Mechanics  
**Task:** Computing expected energy values for quantum harmonic oscillator wavefunctions  

The harmonic oscillator evaluation tests whether language models can compute the expected energy for quantum harmonic oscillator wavefunctions given as linear combinations of energy eigenstates. Models must return exact symbolic energy values (not numerical approximations).

**Key Features:**
- Tests understanding of quantum mechanical principles
- Requires symbolic math manipulation with Hermite polynomials
- Evaluates ability to work with wave functions and energy eigenvalues
- Uses the Verifiers framework for evaluation

**Problem Format:**
- Input: Potential function V(x) = x²/2 and wavefunction ψ(x) as polynomial × exp(-x²/2)
- Output: Exact expected energy as a rational/irrational but not decimal/numerical number (e.g., "207/74")

## Getting Started

Each evaluation subdirectory is self-contained with its own installation and usage instructions. Navigate to the specific evaluation directory and follow the README for setup and execution details.

## Requirements

- Python 3.11+
- uv package manager
- Evaluation-specific dependencies (see individual READMEs)

## Contributing

When adding new physics evaluations:
1. Create a new subdirectory with descriptive name
2. Include comprehensive README with problem description
3. Provide test scripts and example problems
4. Document expected model performance benchmarks