import os

from setuptools import find_packages, setup

_here = os.path.abspath(os.path.dirname(__file__))
_readme = os.path.join(_here, "README.md")
_long_description = ""
if os.path.exists(_readme):
    with open(_readme, encoding="utf-8") as f:
        _long_description = f.read()


QWEN = [
    "transformers==5.4.0",
    "accelerate>=1.12.0",
    "qwen-vl-utils==0.0.14",
    "pillow",
    "numpy>=2.0",
    "einops",
]

# LongVA env (transformers 4.x — a SEPARATE venv from the Qwen extras above;
# the two transformers pins never coexist). The LongVA model package itself is
# NOT on PyPI: clone https://github.com/EvolvingLMMs-Lab/LongVA and
# `pip install -e .` it into this venv. This extras pins the harness around it.
LONGVA = [
    "transformers==4.43.4",
    "accelerate==0.34.2",
    "tokenizers==0.19.1",
    "sentencepiece",
    "decord==0.6.0",
    "av",
    "timm",
    "einops",
    "pillow",
    "numpy>=2.0",
]

setup(
    name="ringzip",
    version="0.1.0",
    description=(
        "RingZip: Coherence-Guided 2D Recursive Token Compression for "
        "Ultra-High-Resolution Remote Sensing Image Understanding with MLLMs"
    ),
    long_description=_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AIRS101/RingZip",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(include=["ringzip", "ringzip.*"]),
    install_requires=["torch>=2.0"],
    extras_require={
        "qwen": QWEN,
        "longva": LONGVA,
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
