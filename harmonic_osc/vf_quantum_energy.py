# vf_quantum_energy.py
# Environment module for willccbb/verifiers
# Generates queries and verifies exact (symbolic) expected energy.

from __future__ import annotations
from typing import Dict, List, Tuple
import json
import random

import sympy as sp
from datasets import Dataset

import verifiers as vf

# -----------------------------
# Problem generator (your "env API")
# -----------------------------
def get_query(
    min_terms: int = 2,
    max_terms: int = 4,
    max_index: int = 6,
    max_coeff_abs: int = 5,
    rng: random.Random | None = None,
) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      {
        "input": {
          "potential": str,   # function of x
          "wavefunction": str # unnormalized linear combination of QHO eigenstates
        },
        "output": {
          "expected_energy": str  # exact sympy expression
        }
      }
    """
    rng = rng or random.Random()

    # 1D harmonic oscillator with ℏ=ω=m=1
    x = sp.Symbol("x")
    potential_expr = sp.Rational(1, 2) * x**2  # V(x) = 1/2 x^2

    k = rng.randint(min_terms, max_terms)
    # sample nonzero integer coeffs and indices
    coeffs: List[int] = []
    indices: List[int] = []
    for _ in range(k):
        c = 0
        while c == 0:
            c = rng.randint(-max_coeff_abs, max_coeff_abs)
        n = rng.randint(0, max_index)
        coeffs.append(c)
        indices.append(n)

    # Build unnormalized wavefunction as a LINEAR COMBINATION of QHO eigenstates
    # For ℏ=ω=m=1, the normalized eigenstate is:
    # ψ_n(x) = (1/√(2^n n!)) * (1/π)^(1/4) * exp(-x^2/2) * H_n(x)
    # We'll create an unnormalized linear combination: Σ c_i * ψ_i(x)
    # Since normalization factors don't affect energy calculation, we can simplify
    
    # Common Gaussian factor
    gaussian = sp.exp(-x**2 / 2)
    
    # Build linear combination of eigenstates (unnormalized)
    terms = []
    for c, n in zip(coeffs, indices):
        # Each term is c_i * exp(-x^2/2) * H_n(x)
        # We can factor out the common Gaussian
        terms.append(sp.Integer(c) * sp.hermite(n, x))
    
    # Full wavefunction: exp(-x^2/2) * Σ c_i * H_n(x)
    psi_expr = gaussian * sum(terms)

    # Expected energy for a linear combination of eigenstates:
    # For normalized wavefunction Ψ = Σ a_i ψ_i where ψ_i are energy eigenstates
    # E = Σ |a_i|^2 E_i / Σ |a_i|^2
    # For QHO with ℏ=ω=1: E_n = n + 1/2
    weights = [abs(c) ** 2 for c in coeffs]
    numer = sum(w * (n + sp.Rational(1, 2)) for w, n in zip(weights, indices))
    denom = sum(weights)
    expected_energy = sp.simplify(sp.Rational(numer, denom))

    # pretty, exact strings
    potential_str = str(potential_expr)
    wavefunction_str = str(psi_expr)
    expected_energy_str = str(expected_energy)

    return {
        "input": {
            "potential": potential_str,
            "wavefunction": wavefunction_str,
        },
        "output": {
            "expected_energy": expected_energy_str
        },
    }


# -----------------------------
# Helper: build a small dataset
# -----------------------------
def _make_dataset(n_examples: int = 200, seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_examples):
        q = get_query(rng=rng)
        # We keep the *ground-truth* only as answer; prompt carries the input.
        prompt = (
            "You are given a 1D quantum harmonic oscillator system with ℏ=ω=m=1.\n"
            "The wavefunction is a linear combination of energy eigenstates.\n\n"
            f'Input:\n{json.dumps({"input": q["input"]}, ensure_ascii=False)}\n\n'
            "Task:\n"
            "Compute the expected energy <E> = <ψ|H|ψ>/<ψ|ψ> for this wavefunction.\n"
            "Return ONLY this JSON with the exact symbolic expression:\n"
            '{"expected_energy": "<sympy expression>"}\n'
        )
        answer = q["output"]["expected_energy"]  # exact expression string
        info = q  # keep the full generated struct for debugging / analysis
        rows.append({"question": prompt, "answer": answer, "info": info})
    return Dataset.from_list(rows)


# -----------------------------
# Robust extraction from model output
# -----------------------------
def _extract_expected_energy_from_completion(completion: List[Dict]) -> str | None:
    """
    Pull the assistant's final message, find JSON, and read expected_energy.
    Returns the raw string expression or None.
    Handles both raw JSON and markdown-wrapped JSON.
    """
    if not completion:
        return None
    # Last assistant message content
    content = ""
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            break
    if not content:
        return None

    # Remove markdown code block markers if present
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove trailing ```
    content = content.strip()

    # Try to locate the first JSON object in the content
    # Simple, tolerant brace matcher
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = content[start : end + 1]

    try:
        obj = json.loads(candidate)
        # Direct access since the expected format is {"expected_energy": "..."}
        val = obj.get("expected_energy")
        if isinstance(val, str):
            return val
        # Sometimes models return numbers instead of strings
        if val is not None:
            return str(val)
        return None
    except Exception:
        return None


# -----------------------------
# Reward: exact (symbolic) match
# -----------------------------
def _energy_exact_match(completion: List[Dict], answer: str, **_) -> float:
    """
    1. Parse model JSON.
    2. Sympify both strings; reward 1.0 iff they are symbolically equal.
    """
    pred = _extract_expected_energy_from_completion(completion)
    if pred is None:
        return 0.0
    try:
        lhs = sp.simplify(sp.sympify(pred))
        rhs = sp.simplify(sp.sympify(answer))
        ok = sp.simplify(lhs - rhs) == 0
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0


# -----------------------------
# Public entry point
# -----------------------------
def load_environment(
    num_examples: int = 200,
    seed: int = 0,
    **kwargs,
) -> vf.SingleTurnEnv:
    """
    Create a SingleTurnEnv that:
      - Prompts for JSON: {"expected_energy": "<sympy expr>"}
      - Verifies exact equality symbolically (rational/irrational forms OK, numerical/decimal forms not OK)
    """
    dataset = _make_dataset(n_examples=num_examples, seed=seed)

    # Strict output format instruction (since we use `question`, not `prompt`)
    system_prompt = (
        "Respond with ONLY a single JSON object matching the schema:\n"
        '{"expected_energy": "<value>"}\n'
        "Where <value> is a string containing the sympy expression.\n"
        "Do not include markdown code blocks, explanations, or any other text.\n"
        "Output only the raw JSON object."
    )

    rubric = vf.Rubric(funcs=[_energy_exact_match])  # 0/1 reward only

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
