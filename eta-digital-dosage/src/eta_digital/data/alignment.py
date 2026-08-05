from __future__ import annotations
import pandas as pd

def align_process_data(inputs:pd.DataFrame,outputs:pd.DataFrame,delay_minutes:int,tolerance_minutes:int=10)->pd.DataFrame:
    if "timestamp" not in inputs or "timestamp" not in outputs: raise ValueError("both frames require timestamp")
    left=inputs.copy(); right=outputs.copy(); left["target_timestamp"]=pd.to_datetime(left["timestamp"],utc=True)+pd.to_timedelta(delay_minutes,unit="m"); right["timestamp"]=pd.to_datetime(right["timestamp"],utc=True)
    left=left.sort_values("target_timestamp"); right=right.sort_values("timestamp")
    return pd.merge_asof(left,right,left_on="target_timestamp",right_on="timestamp",direction="nearest",tolerance=pd.to_timedelta(tolerance_minutes,unit="m"),suffixes=("_input","_output"))
