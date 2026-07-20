"""
BaseMetric
==========
Abstract base class that every metric in the pipeline must subclass.

Contract
--------
• Subclasses set class-level `name` (str) and `phase` (int).
• `run(**kwargs) → dict` is the only required method.  The returned dict
  must be JSON-serialisable.  All heavy computation lives here.
• Metrics are stateless — create a new instance per call if needed, or
  reuse the same instance safely because run() has no side-effects on self.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetric(ABC):
    name: str = ""   # human-readable identifier used as the JSON key
    phase: int = 0   # pipeline phase this metric belongs to (1, 2, or 3)

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Compute the metric and return a plain, JSON-serialisable dict.

        Keyword arguments vary per metric — see each subclass for the
        specific signature.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, phase={self.phase})"
