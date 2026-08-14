import math
from pydantic import BaseModel, ConfigDict, Field
from typing import Union

# Define a strict numeric type that disallows NaN/Inf
StrictNumeric = float | int

class TransactionRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    
    Time: StrictNumeric = Field(..., allow_inf_nan=False)
    Amount: StrictNumeric = Field(..., allow_inf_nan=False)
    V1: StrictNumeric = Field(..., allow_inf_nan=False)
    V2: StrictNumeric = Field(..., allow_inf_nan=False)
    V3: StrictNumeric = Field(..., allow_inf_nan=False)
    V4: StrictNumeric = Field(..., allow_inf_nan=False)
    V5: StrictNumeric = Field(..., allow_inf_nan=False)
    V6: StrictNumeric = Field(..., allow_inf_nan=False)
    V7: StrictNumeric = Field(..., allow_inf_nan=False)
    V8: StrictNumeric = Field(..., allow_inf_nan=False)
    V9: StrictNumeric = Field(..., allow_inf_nan=False)
    V10: StrictNumeric = Field(..., allow_inf_nan=False)
    V11: StrictNumeric = Field(..., allow_inf_nan=False)
    V12: StrictNumeric = Field(..., allow_inf_nan=False)
    V13: StrictNumeric = Field(..., allow_inf_nan=False)
    V14: StrictNumeric = Field(..., allow_inf_nan=False)
    V15: StrictNumeric = Field(..., allow_inf_nan=False)
    V16: StrictNumeric = Field(..., allow_inf_nan=False)
    V17: StrictNumeric = Field(..., allow_inf_nan=False)
    V18: StrictNumeric = Field(..., allow_inf_nan=False)
    V19: StrictNumeric = Field(..., allow_inf_nan=False)
    V20: StrictNumeric = Field(..., allow_inf_nan=False)
    V21: StrictNumeric = Field(..., allow_inf_nan=False)
    V22: StrictNumeric = Field(..., allow_inf_nan=False)
    V23: StrictNumeric = Field(..., allow_inf_nan=False)
    V24: StrictNumeric = Field(..., allow_inf_nan=False)
    V25: StrictNumeric = Field(..., allow_inf_nan=False)
    V26: StrictNumeric = Field(..., allow_inf_nan=False)
    V27: StrictNumeric = Field(..., allow_inf_nan=False)
    V28: StrictNumeric = Field(..., allow_inf_nan=False)

class PredictionResponse(BaseModel):
    prediction: str
    is_fraud: bool
    fraud_probability: float
    threshold: float
    model_name: str
