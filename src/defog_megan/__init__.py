"""Concept-conditioned graph generation on top of defog-adapt.

Generate graphs for a requested property and for the structural concept through which a graph reaches it: the
condition is the concept-membership vector of a self-explaining GNN, the generator a frozen DeFoG model steered by
adapters and Feynman-Kac particle steering. This is a pre-alpha skeleton; see the README for what exists.
"""
from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("defog-megan")
except _metadata.PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
