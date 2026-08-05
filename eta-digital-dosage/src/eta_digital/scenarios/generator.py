from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd
from eta_digital.experts.mixture import ContextualMixtureOfExperts

@dataclass
class ScenarioSet:
    outcomes: np.ndarray
    weights: np.ndarray
    output_names: list[str]

class WeightedScenarioGenerator:
    def __init__(self,predictor:ContextualMixtureOfExperts,number_of_scenarios:int=400,sensor_std:Mapping[str,float]|None=None,random_state:int|None=42):
        if number_of_scenarios<20: raise ValueError("at least 20 scenarios are required")
        self.predictor=predictor; self.number_of_scenarios=number_of_scenarios; self.sensor_std=dict(sensor_std or {}); self.rng=np.random.default_rng(random_state)
    def generate(self,state:pd.DataFrame)->ScenarioSet:
        if len(state)!=1: raise ValueError("scenario generation expects one operating state")
        repeated=pd.concat([state]*self.number_of_scenarios,ignore_index=True)
        for name,std in self.sensor_std.items():
            if name in repeated: repeated[name]=repeated[name].to_numpy(float)+self.rng.normal(0,float(std),len(repeated))
        distribution=self.predictor.predict_distribution(repeated)
        samples=np.empty_like(distribution.mean.to_numpy(float))
        for i,(mean,cov) in enumerate(zip(distribution.mean.to_numpy(float),distribution.covariance)):
            samples[i]=self.rng.multivariate_normal(mean,cov,check_valid="warn")
        return ScenarioSet(samples,np.full(self.number_of_scenarios,1/self.number_of_scenarios),self.predictor.output_features)
