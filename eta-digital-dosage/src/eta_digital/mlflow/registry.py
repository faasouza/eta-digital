from __future__ import annotations


def register_prediction_model(
    model,
    artifact_path: str,
    registered_model_name: str,
    input_example,
):
    import mlflow
    from mlflow.models import infer_signature

    output_example = model.predict(input_example)
    signature = infer_signature(input_example, output_example)
    return mlflow.pyfunc.log_model(
        name=artifact_path,
        python_model=model,
        input_example=input_example,
        signature=signature,
        registered_model_name=registered_model_name,
    )


def set_model_alias(model_name: str, version: str, alias: str = "champion") -> None:
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(model_name, alias, version)
