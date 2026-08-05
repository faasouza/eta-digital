from __future__ import annotations
from pathlib import Path
from typing import Any
from .pyfunc_model import EtaDigitalPredictionModel
from .signatures import build_input_example,infer_prediction_signature

def _mlflow():
    try: import mlflow
    except ImportError as exc: raise RuntimeError("MLflow is required") from exc
    return mlflow

def configure_tracking(tracking_uri:str,experiment_name:str):
    m=_mlflow(); m.set_tracking_uri(tracking_uri); m.set_experiment(experiment_name)

def log_and_register_prediction_model(predictor,*,registered_model_name:str,calibrator=None,run_name=None,metrics:dict[str,float]|None=None,parameters:dict[str,Any]|None=None,candidate_alias:str="candidate"):
    m=_mlflow(); wrapper=EtaDigitalPredictionModel(predictor,calibrator); x=build_input_example(); sig=infer_prediction_signature(wrapper)
    with m.start_run(run_name=run_name) as run:
        if parameters: m.log_params({k:str(v) for k,v in parameters.items()})
        if metrics: m.log_metrics({k:float(v) for k,v in metrics.items()})
        info=m.pyfunc.log_model(name="prediction_model",python_model=wrapper,registered_model_name=registered_model_name,signature=sig,input_example=x,code_paths=[str(Path(__file__).resolve().parents[2])],pip_requirements=["numpy>=1.26,<3","pandas>=2.1,<3","scikit-learn>=1.4,<2"],metadata={"component":"contextual_prediction","optimization_included":False,"supervision_included":False})
        run_id=run.info.run_id
    client=m.MlflowClient(); versions=[v for v in client.search_model_versions(f"name='{registered_model_name}'") if v.run_id==run_id]
    if not versions: raise RuntimeError("registered model version not found")
    version=str(max(versions,key=lambda v:int(v.version)).version); client.set_registered_model_alias(registered_model_name,candidate_alias,version); client.set_model_version_tag(registered_model_name,version,"validation_status","candidate")
    return {"run_id":run_id,"model_uri":info.model_uri,"version":version,"alias":candidate_alias}

def promote_model_version(registered_model_name:str,version:str|int,*,validation_passed:bool,production_alias:str="champion",validation_summary:str=""):
    if not validation_passed: raise ValueError("promotion requires validation_passed=True")
    m=_mlflow(); client=m.MlflowClient(); v=str(version); client.set_model_version_tag(registered_model_name,v,"validation_status","passed")
    if validation_summary: client.set_model_version_tag(registered_model_name,v,"validation_summary",validation_summary)
    client.set_registered_model_alias(registered_model_name,production_alias,v)
def load_model_by_alias(registered_model_name:str,alias:str="champion"):
    return _mlflow().pyfunc.load_model(f"models:/{registered_model_name}@{alias}")
