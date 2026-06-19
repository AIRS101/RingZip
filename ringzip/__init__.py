"""RingZip: Coherence-Guided 2D Recursive Token Compression for UHR RS MLLMs.

RingZip compresses the 2D visual-token grid produced by an MLLM's vision
encoder *at the region level*. It estimates a region-wise weighted coherence

    C(B) = || sum_{i in B} f_i || / sum_{i in B} || f_i ||

and runs a coarse-to-fine recursive split-and-merge over the grid, collapsing
large homogeneous regions into single tokens while preserving low-coherence
detail and the 2D spatial topology.

Public API
----------
``RingZipCompressor``
    Builds a :class:`CompressPlan` from ViT-side 2D token features and applies
    it to hidden states / position ids / KV cache.
``CompressPlan``
    The resulting input-token -> output-region mapping.

Example
-------
>>> import torch
>>> from ringzip import RingZipCompressor
>>> vis = torch.randn(32 * 32, 1024)        # (N, D) ViT tokens on a 32x32 grid
>>> comp = RingZipCompressor(init_stride=16, norm_temperature=1 / 3)
>>> plan = comp.plan_compression(vis, h_tok=32, w_tok=32)
>>> out = comp.compress_hidden(vis, plan)   # (M, D), M <= N
"""

from .core import CompressPlan, RingZipCompressor

# Alias keeping the paper's wording for the plan object.
RingZipPlan = CompressPlan

__all__ = [
    "RingZipCompressor",
    "RingZipPlan",
    "CompressPlan",
]

__version__ = "0.1.0"
