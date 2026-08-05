from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

class MembershipFunction(Protocol):
    def __call__(self, values: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class TriangularMembership:
    a: float; b: float; c: float
    def __post_init__(self):
        if not self.a <= self.b <= self.c or self.a == self.c:
            raise ValueError("require a <= b <= c and a < c")
    def __call__(self, values: np.ndarray) -> np.ndarray:
        x=np.asarray(values,float); out=np.zeros_like(x)
        if self.b>self.a:
            mask=(x>=self.a)&(x<=self.b); out[mask]=(x[mask]-self.a)/(self.b-self.a)
        else: out[x==self.a]=1
        if self.c>self.b:
            mask=(x>=self.b)&(x<=self.c); out[mask]=np.maximum(out[mask],(self.c-x[mask])/(self.c-self.b))
        else: out[x==self.c]=1
        out[x==self.b]=1
        return np.clip(out,0,1)

@dataclass(frozen=True)
class TrapezoidalMembership:
    a: float; b: float; c: float; d: float
    def __post_init__(self):
        if not self.a <= self.b <= self.c <= self.d or self.a == self.d:
            raise ValueError("require a <= b <= c <= d and a < d")
    def __call__(self, values: np.ndarray) -> np.ndarray:
        x=np.asarray(values,float); out=np.zeros_like(x)
        plateau=(x>=self.b)&(x<=self.c); out[plateau]=1
        if self.b>self.a:
            m=(x>=self.a)&(x<self.b); out[m]=(x[m]-self.a)/(self.b-self.a)
        else: out[x==self.a]=1
        if self.d>self.c:
            m=(x>self.c)&(x<=self.d); out[m]=(self.d-x[m])/(self.d-self.c)
        else: out[x==self.d]=1
        return np.clip(out,0,1)

@dataclass(frozen=True)
class GaussianMembership:
    mean: float; sigma: float
    def __post_init__(self):
        if self.sigma<=0: raise ValueError("sigma must be positive")
    def __call__(self, values: np.ndarray) -> np.ndarray:
        x=np.asarray(values,float); return np.exp(-0.5*((x-self.mean)/self.sigma)**2)

def build_membership(kind: str, parameters: Sequence[float]) -> MembershipFunction:
    values=[float(v) for v in parameters]
    if kind=="triangular": return TriangularMembership(*values)
    if kind=="trapezoidal": return TrapezoidalMembership(*values)
    if kind=="gaussian": return GaussianMembership(*values)
    raise ValueError(f"unsupported membership type: {kind}")
