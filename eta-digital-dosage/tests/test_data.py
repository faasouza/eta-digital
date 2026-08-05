import pandas as pd
from eta_digital.data.alignment import align_process_data
from eta_digital.data.validation import validate_training_frame

def test_alignment_and_validation(training_frame):
    inp=pd.DataFrame({"timestamp":["2026-01-01T00:00:00Z"],"x":[1]}); out=pd.DataFrame({"timestamp":["2026-01-01T00:30:00Z"],"y":[2]}); aligned=align_process_data(inp,out,30,1); assert aligned.y.iloc[0]==2; assert len(validate_training_frame(training_frame))==len(training_frame)
