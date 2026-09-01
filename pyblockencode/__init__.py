"""
pyblockencode — structured block encodings of finite element operators.

Shift decomposition writes each 1D finite element factor as a short sum of
cyclic shift operators, giving a linear combination of unitaries whose term
count L and subnormalization alpha are both constant in the grid size N.
Boundary conditions are imposed by adjoining reflection operators to that
unitary set, which costs no ancilla.

    from pyblockencode import PoissonEncoding, ElasticityEncoding

    enc = ElasticityEncoding(m=3, nu=0.3, bc='essential')
    enc.num_terms            # 49
    enc.alpha                # 6.099, constant in m
    enc.verify()             # error metrics against a direct Q4 assembly

The circuit layer needs qiskit and is imported on demand, so the algebraic
layer works without it:

    from pyblockencode import ElasticityCircuit          # requires qiskit

Modules
-------
bc                  the unitary set, the 1D factors, boundary treatments
operators           classically assembled reference operators
poisson_pattern     PoissonEncoding
elasticity_pattern  ElasticityEncoding
qiskit_encoding     PauliBlockEncoding, PoissonCircuit, ElasticityCircuit
linear_circuits     the same circuits with an O(m)-Toffoli incrementer

The verification harness lives at the repository root: ``python smoke_test.py``.
"""
from __future__ import annotations

from . import bc, operators
from .elasticity_pattern import ElasticityEncoding, ElasticityPatternEncoding
from .poisson_pattern import PoissonEncoding, PoissonPatternEncoding

__all__ = [
    "bc", "operators",
    "PoissonEncoding", "ElasticityEncoding",
    # deprecated aliases, kept so the published API still resolves
    "PoissonPatternEncoding", "ElasticityPatternEncoding",
    # qiskit-backed, resolved lazily by __getattr__ below
    "PauliBlockEncoding", "PoissonCircuit", "ElasticityCircuit", "apply_Ax",
    "LinearPoissonCircuit", "LinearElasticityCircuit", "verify_columns",
]

_LAZY = {
    "PauliBlockEncoding": "qiskit_encoding",
    "PoissonCircuit": "qiskit_encoding",
    "ElasticityCircuit": "qiskit_encoding",
    "apply_Ax": "qiskit_encoding",
    "LinearPoissonCircuit": "linear_circuits",
    "LinearElasticityCircuit": "linear_circuits",
    "verify_columns": "linear_circuits",
}


def __getattr__(name: str):
    """Import the circuit classes only when they are asked for.

    Keeping qiskit off the import path means the algebraic layer -- which is
    the part the paper's results rest on -- runs with numpy alone.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        mod = importlib.import_module(f".{module}", __name__)
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            f"{name} lives in pyblockencode.{module}, which needs qiskit: "
            f"pip install qiskit qiskit-aer"
        ) from exc
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
