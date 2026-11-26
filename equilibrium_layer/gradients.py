"""Parameter VJP container and weighted aggregation arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .parameters import Parameters


@dataclass(frozen=True)
class ParameterGradients:
    W: np.ndarray
    U: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        W = np.asarray(self.W, dtype=np.float64)
        U = np.asarray(self.U, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        object.__setattr__(self, "W", W)
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "b", b)

    @classmethod
    def zeros_like(cls, params: Parameters) -> "ParameterGradients":
        return cls(np.zeros_like(params.W), np.zeros_like(params.U), np.zeros_like(params.b))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ParameterGradients":
        return cls(data["W"], data["U"], data["b"])

    def __add__(self, other: "ParameterGradients") -> "ParameterGradients":
        if not isinstance(other, ParameterGradients):
            return NotImplemented
        return ParameterGradients(self.W + other.W, self.U + other.U, self.b + other.b)

    def __mul__(self, weight: float) -> "ParameterGradients":
        weight = float(weight)
        return ParameterGradients(self.W * weight, self.U * weight, self.b * weight)

    __rmul__ = __mul__

    def __truediv__(self, denominator: float) -> "ParameterGradients":
        denominator = float(denominator)
        if denominator <= 0.0:
            raise ZeroDivisionError("effective eligible sample weight must be positive")
        return ParameterGradients(self.W / denominator, self.U / denominator, self.b / denominator)

    def max_abs(self) -> float:
        return max(float(np.max(np.abs(self.W))), float(np.max(np.abs(self.U))), float(np.max(np.abs(self.b))))

    def relative_difference(self, other: "ParameterGradients") -> dict[str, float]:
        out: dict[str, float] = {}
        for name in ("W", "U", "b"):
            a = getattr(self, name)
            b = getattr(other, name)
            denom = max(float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel())), 1e-12)
            out[name] = float(np.linalg.norm((a - b).ravel()) / denom)
        out["max"] = max(out["W"], out["U"], out["b"])
        return out

    def to_dict(self) -> dict[str, list]:
        return {"W": self.W.tolist(), "U": self.U.tolist(), "b": self.b.tolist()}
