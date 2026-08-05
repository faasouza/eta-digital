from pydantic import BaseModel, Field

class ProcessRecord(BaseModel):
    raw_turbidity_ntu:float=Field(ge=0); raw_ph:float=Field(ge=0,le=14); flow_m3_h:float=Field(gt=0); temperature_c:float=Field(ge=0,le=60); pac_mg_l:float=Field(ge=0); polymer_mg_l:float=Field(ge=0)
class PredictionRequest(ProcessRecord): pass
class RecommendationRequest(BaseModel):
    raw_turbidity_ntu:float=Field(ge=0); raw_ph:float=Field(ge=0,le=14); flow_m3_h:float=Field(gt=0); temperature_c:float=Field(ge=0,le=60)
    previous_pac_mg_l:float=Field(ge=0); previous_polymer_mg_l:float=Field(ge=0); sensor_quality:float=Field(default=1,ge=0,le=1); out_of_domain_score:float=Field(default=0,ge=0,le=1)
