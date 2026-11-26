"""Shared model parameters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from .errors import ValidationError


@dataclass(frozen=True)
class Parameters:
    W: np.ndarray
    U: np.ndarray
    b: np.ndarray
    generation: int = 0
    parameter_id: str | None = None

    def __post_init__(self) -> None:
        W = np.asarray(self.W, dtype=np.float64).copy()
        U = np.asarray(self.U, dtype=np.float64).copy()
        b = np.asarray(self.b, dtype=np.float64).copy()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValidationError(f"W must be square 2D, got {W.shape}")
        d = W.shape[0]
        if U.ndim != 2 or U.shape[1] != d:
            raise ValidationError(f"U must have shape (input_dim, {d}), got {U.shape}")
        if b.shape != (d,):
            raise ValidationError(f"b must have shape ({d},), got {b.shape}")
        for name, a in (("W", W), ("U", U), ("b", b)):
            if not np.all(np.isfinite(a)):
                raise ValidationError(f"parameter {name} contains non-finite values")
        object.__setattr__(self, "W", W)
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "generation", int(self.generation))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, generation: int = 0) -> "Parameters":
        return cls(
            W=data["W"],
            U=data["U"],
            b=data["b"],
            generation=int(data.get("generation", generation)),
            parameter_id=data.get("id") or data.get("parameter_id"),
        )

    @property
    def state_dim(self) -> int:
        return int(self.W.shape[0])

    def snapshot(self) -> "Parameters":
        """Return an isolated copy whose arrays cannot alias caller storage."""
        return Parameters(
            W=self.W, U=self.U, b=self.b,
            generation=self.generation, parameter_id=self.parameter_id,
        )

    @property
    def input_dim(self) -> int:
        return int(self.U.shape[0])

    def next_generation(self, **arrays: Any) -> "Parameters":
        return Parameters(
            W=arrays.get("W", self.W),
            U=arrays.get("U", self.U),
            b=arrays.get("b", self.b),
            generation=self.generation + 1,
            parameter_id=arrays.get("parameter_id"),
        )

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(b"qeq-params-v1")
        h.update(str(self.generation).encode())
        for name, a in (("W", self.W), ("U", self.U), ("b", self.b)):
            h.update(name.encode())
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
