from eta_digital.mlflow import EtaDigitalPredictionModel
from test_mixture import build_model


def test_pyfunc_wrapper_predicts_new_schema_without_mlflow_server():
    model, frame = build_model()
    wrapper = EtaDigitalPredictionModel(model)
    result = wrapper.predict(None, frame.iloc[200:203][model.features])
    assert set(model.outputs).issubset(result.columns)
    assert len(result) == 3
