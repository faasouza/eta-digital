from eta_digital.mlflow import EtaDigitalPredictionModel
from test_mixture import build_model


def test_pyfunc_wrapper_predicts_new_schema_without_mlflow_server():
    model, frame = build_model()
    wrapper = EtaDigitalPredictionModel(model)
    result = wrapper.predict(None, frame.iloc[200:203][model.features])
    assert set(model.outputs).issubset(result.columns)
    assert len(result) == 3


def test_registry_calls_pyfunc_with_context_and_logs_model(monkeypatch):
    import sys
    import types

    from eta_digital.mlflow.registry import register_prediction_model
    from eta_digital.mlflow.signatures import prediction_input_example

    model, _ = build_model()
    wrapper = EtaDigitalPredictionModel(model)
    recorded = {}

    fake_models = types.ModuleType("mlflow.models")
    fake_models.infer_signature = lambda inputs, outputs: (
        tuple(inputs.columns), tuple(outputs.columns)
    )

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.models = fake_models
    fake_mlflow.pyfunc = types.SimpleNamespace(
        log_model=lambda **kwargs: recorded.update(kwargs)
        or types.SimpleNamespace(model_uri="models:/test/1")
    )

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", fake_models)

    result = register_prediction_model(
        wrapper,
        artifact_path="prediction_model",
        registered_model_name="eta-digital-test",
        input_example=prediction_input_example(),
    )

    assert result.model_uri == "models:/test/1"
    assert recorded["name"] == "prediction_model"
    assert recorded["registered_model_name"] == "eta-digital-test"
    assert set(recorded["input_example"].columns) == set(model.features)
