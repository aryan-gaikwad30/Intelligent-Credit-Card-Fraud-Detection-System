import pandas as pd
from backend.app.schemas.predict import TransactionRequest, PredictionResponse
from backend.app.ml.feature_engineering import FeatureEngineer

# Initialize a single FeatureEngineer instance for reuse
feature_engineer = FeatureEngineer()
# Need to set is_fitted_ to True since we bypass fit()
feature_engineer.is_fitted_ = True

def predict_fraud(transaction: TransactionRequest, model, threshold: float, feature_cols: list, model_name: str) -> PredictionResponse:
    # 1. Convert request to DataFrame
    transaction_dict = transaction.model_dump()
    df = pd.DataFrame([transaction_dict])
    
    # 2. Apply deterministic feature engineering
    # The FeatureEngineer transforms Time -> time_of_day_sin/cos and Amount -> amount_log1p
    df_transformed = feature_engineer.transform(df)
    
    # 3. Align columns to match the exact training feature order
    # Any missing columns (e.g. if we somehow didn't generate them) will raise a KeyError
    X = df_transformed[feature_cols]
    
    # 4. Predict probability
    probs = model.predict_proba(X)
    fraud_prob = float(probs[0, 1])
    
    # 5. Apply locked threshold
    is_fraud = fraud_prob >= threshold
    prediction_label = "fraud" if is_fraud else "legitimate"
    
    return PredictionResponse(
        prediction=prediction_label,
        is_fraud=is_fraud,
        fraud_probability=fraud_prob,
        threshold=threshold,
        model_name=model_name
    )
