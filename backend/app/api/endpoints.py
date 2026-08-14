from fastapi import APIRouter, Request, HTTPException
from backend.app.schemas.predict import TransactionRequest, PredictionResponse
from backend.app.services.inference import predict_fraud
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check(request: Request):
    """
    Check if the API is running and the model is loaded.
    """
    model_loaded = False
    if hasattr(request.app.state, "model") and request.app.state.model is not None:
        model_loaded = True
        
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest, request: Request):
    """
    Predict fraud probability for a single transaction.
    """
    # Verify model is loaded
    if not hasattr(request.app.state, "model") or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        model = request.app.state.model
        threshold = request.app.state.threshold
        feature_columns = request.app.state.feature_columns
        model_name = request.app.state.model_name
        
        response = predict_fraud(
            transaction=transaction,
            model=model,
            threshold=threshold,
            feature_cols=feature_columns,
            model_name=model_name
        )
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Do not leak stack traces to the client
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
