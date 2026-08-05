from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .membership import MembershipFunction, build_membership


@dataclass(frozen=True)
class ContextDefinition:
    name: str
    memberships: dict[str, MembershipFunction]


class ContextModel:
    def __init__(self, contexts: list[ContextDefinition], minimum_weight: float = 1e-6):
        if not contexts:
            raise ValueError("at least one context is required")
        self.contexts = contexts
        self.minimum_weight = float(minimum_weight)

    @classmethod
    def from_config(cls, config: dict, minimum_weight: float = 1e-6) -> "ContextModel":
        contexts = []
        for name, variables in config["contexts"].items():
            memberships = {
                variable: build_membership(spec["type"], list(spec["parameters"]))
                for variable, spec in variables.items()
            }
            contexts.append(ContextDefinition(name=name, memberships=memberships))
        return cls(contexts, minimum_weight=minimum_weight)

    @property
    def names(self) -> list[str]:
        return [context.name for context in self.contexts]

    def raw_possibilities(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.empty((len(frame), len(self.contexts)), dtype=float)
        for context_index, context in enumerate(self.contexts):
            activation = np.ones(len(frame), dtype=float)
            for variable, membership in context.memberships.items():
                if variable not in frame:
                    raise ValueError(f"missing context variable: {variable}")
                activation = np.minimum(activation, membership(frame[variable].to_numpy()))
            values[:, context_index] = activation
        return values

    def weights(self, frame: pd.DataFrame) -> np.ndarray:
        possibilities = np.maximum(self.raw_possibilities(frame), self.minimum_weight)
        totals = possibilities.sum(axis=1, keepdims=True)
        return possibilities / totals
