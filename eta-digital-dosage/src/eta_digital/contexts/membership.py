from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MembershipFunction(Protocol):
    def __call__(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TriangularMembership:
    a: float
    b: float
    c: float

    def __post_init__(self) -> None:
        if not self.a <= self.b <= self.c or self.a == self.c:
            raise ValueError("triangular parameters must satisfy a <= b <= c and a < c")

    def __call__(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        left = np.ones_like(x) if self.a == self.b else (x - self.a) / (self.b - self.a)
        right = np.ones_like(x) if self.b == self.c else (self.c - x) / (self.c - self.b)
        return np.clip(np.minimum(left, right), 0.0, 1.0)


@dataclass(frozen=True)
class TrapezoidalMembership:
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if not self.a <= self.b <= self.c <= self.d or self.a == self.d:
            raise ValueError("trapezoidal parameters must satisfy a <= b <= c <= d and a < d")

    def __call__(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        rise = np.ones_like(x) if self.a == self.b else (x - self.a) / (self.b - self.a)
        fall = np.ones_like(x) if self.c == self.d else (self.d - x) / (self.d - self.c)
        return np.clip(np.minimum(np.minimum(rise, 1.0), fall), 0.0, 1.0)


@dataclass(frozen=True)
class GaussianMembership:
    center: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def __call__(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)


def build_membership(kind: str, parameters: list[float]) -> MembershipFunction:
    normalized = kind.strip().lower()
    if normalized == "triangular":
        return TriangularMembership(*parameters)
    if normalized == "trapezoidal":
        return TrapezoidalMembership(*parameters)
    if normalized == "gaussian":
        return GaussianMembership(*parameters)
    raise ValueError(f"unsupported membership type: {kind}")
