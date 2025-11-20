"""Contextual bandit data generating processes."""

from .base import DGPBase
from .simple import Simple

__all__ = ["DGPBase", "Simple"]
