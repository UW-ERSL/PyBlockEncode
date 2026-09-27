from setuptools import setup

setup(
    name="pyblockencode",
    version="0.1.0",
    description="Block encodings of periodic finite-element operators",
    url="https://github.com/UW-ERSL/PyBlockEncode",
    py_modules=["pyblockencode"],
    python_requires=">=3.10",
    install_requires=["numpy", "qiskit>=1.0"],
)