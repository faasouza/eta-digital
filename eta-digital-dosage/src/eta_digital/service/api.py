from __future__ import annotations
import pandas as pd
from fastapi import FastAPI,HTTPException
from eta_digital.data.schemas import PredictionRequest,RecommendationRequest

class Runtime:
    predictor=None; optimizer=None; supervisor=None

def create_app(predictor=None,optimizer=None,supervisor=None)->FastAPI:
    app=FastAPI(title="ETA-DIGITAL Dosage API",version="0.1.0")
    runtime=Runtime(); runtime.predictor=predictor; runtime.optimizer=optimizer; runtime.supervisor=supervisor
    @app.get("/health")
    def health(): return {"status":"ok","predictor_loaded":runtime.predictor is not None}
    @app.post("/predict")
    def predict(request:PredictionRequest):
        if runtime.predictor is None: raise HTTPException(503,"predictor not loaded")
        frame=pd.DataFrame([request.model_dump()]); result=runtime.predictor.predict_distribution(frame)
        return {"prediction":result.mean.iloc[0].to_dict(),"context_confidence":float(result.context_confidence.iloc[0]),"dominant_context":str(result.dominant_context.iloc[0]),"context_weights":result.context_weights.iloc[0].to_dict()}
    @app.post("/recommend")
    def recommend(request:RecommendationRequest):
        if runtime.optimizer is None or runtime.supervisor is None: raise HTTPException(503,"decision components not loaded")
        data=request.model_dump(); previous=(data.pop("previous_pac_mg_l"),data.pop("previous_polymer_mg_l")); sensor_quality=data.pop("sensor_quality"); ood=data.pop("out_of_domain_score")
        frame=pd.DataFrame([{**data,"pac_mg_l":previous[0],"polymer_mg_l":previous[1]}]); result=runtime.optimizer.solve(frame,previous)
        dist=runtime.optimizer.generator.predictor.predict_distribution(frame); decision=runtime.supervisor.evaluate(result,float(dist.context_confidence.iloc[0]),sensor_quality,ood,previous)
        return {"pac_mg_l":decision.pac_mg_l,"polymer_mg_l":decision.polymer_mg_l,"status":decision.status,"confidence":decision.confidence,"reason":decision.reason,"optimization":{"objective":result.objective,"compliance_probability":result.compliance_probability,"feasible":result.feasible}}
    return app
