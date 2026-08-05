from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PredictionDistribution:
    mean: pd.DataFrame
    covariance: np.ndarray
    context_weights: pd.DataFrame
    context_confidence: pd.Series
    dominant_context: pd.Series
