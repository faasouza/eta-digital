import numpy as np

def test_mixture_predictions_and_covariance(predictor,training_frame):
    result=predictor.predict_distribution(training_frame.head(5)); assert result.mean.shape==(5,2); assert result.covariance.shape==(5,2,2); assert np.all(np.linalg.eigvalsh(result.covariance)>=-1e-8)
