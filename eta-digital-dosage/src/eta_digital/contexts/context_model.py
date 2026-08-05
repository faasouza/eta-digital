from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Mapping
import numpy as np
import pandas as pd
from .membership import MembershipFunction, build_membership

@dataclass(frozen=True)
class ContextDefinition:
    name: str
    memberships: Mapping[str, MembershipFunction]

class ContextModel:
    def __init__(self, contexts: list[ContextDefinition], epsilon: float=1e-12):
        if not contexts: raise ValueError("at least one context is required")
        self.contexts=contexts; self.epsilon=float(epsilon)
    @property
    def context_names(self): return [c.name for c in self.contexts]
    @classmethod
    def from_config(cls, config: Mapping):
        definitions=[]
        for name, variables in config["contexts"].items():
            memberships={feature: build_membership(spec["type"], spec["parameters"]) for feature,spec in variables.items()}
            definitions.append(ContextDefinition(name,memberships))
        return cls(definitions)
    def possibilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        out={}
        for context in self.contexts:
            missing=set(context.memberships)-set(frame.columns)
            if missing: raise ValueError(f"missing context features: {sorted(missing)}")
            parts=[fn(frame[name].to_numpy(float)) for name,fn in context.memberships.items()]
            out[context.name]=reduce(mul,parts,np.ones(len(frame)))
        return pd.DataFrame(out,index=frame.index)
    def weights(self, frame: pd.DataFrame) -> pd.DataFrame:
        poss=self.possibilities(frame)
        sums=poss.sum(axis=1)
        zero=sums<=self.epsilon
        weights=poss.div(sums.where(~zero,1.0),axis=0)
        if zero.any(): weights.loc[zero,:]=1.0/len(self.contexts)
        return weights
