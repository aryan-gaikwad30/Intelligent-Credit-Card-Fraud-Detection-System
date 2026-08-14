# FastAPI Model Serving Documentation

This document describes the Phase 8 API implementation for the Intelligent Credit Card Fraud Detection System.

## Architecture Overview

The API is built using **FastAPI** to expose the locked XGBoost Phase 4 Baseline model.
It securely relies on the `FeatureEngineer` from `backend/app/ml/feature_engineering.py` for all deterministic transformations (`time_of_day_sin/cos`, `amount_log1p`), avoiding code duplication.

## Configuration & Model Loading

- The model is strictly bound to `models/final_model_config.json`.
- The configuration requires `threshold = 0.31` and `model_name = "Phase 4 XGBoost Baseline"`.
- The model is loaded exactly once into `app.state` using a FastAPI lifespan context manager to prevent performance degradation on individual predictions.
- If the required config values do not align, the application throws a fatal startup exception to protect production logic.

## Endpoints

### 1. `GET /api/v1/health`
Checks whether the API is alive and reports model loading status.

**Example Request:**
```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Example Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. `POST /api/v1/predict`
Runs inference on a single transaction payload.

**Request Schema (`TransactionRequest`):**
Requires `Time`, `Amount`, and `V1`...`V28` as strict floats/ints. 
NaN, +Infinity, and -Infinity are explicitly rejected to prevent corrupted inference.
If a field is missing, Pydantic immediately responds with a `422 Unprocessable Entity`.

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
-H "Content-Type: application/json" \
-d '{
  "Time": 0.0,
  "Amount": 100.0,
  "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
  "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
  "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
  "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
  "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
  "V26": 2.6, "V27": 2.7, "V28": 2.8
}'
```

**Example Response:**
```json
{
  "prediction": "legitimate",
  "is_fraud": false,
  "fraud_probability": 0.0104,
  "threshold": 0.31,
  "model_name": "Phase 4 XGBoost Baseline"
}
```

## Validation & Error Handling
1. **Pydantic Validation**: Catches string coercion errors and missing keys (`422 Unprocessable Entity`).
2. **NaN/Inf Validator**: Specifically identifies and rejects invalid scientific bounds to protect the ML model (`422 Unprocessable Entity`).
3. **Internal Server Errors**: The `/predict` endpoint wraps inference inside a generic exception handler. Standard 500 errors are returned (`500 Internal Server Error`) while stack traces are logged but not exposed to users.
4. **Model Loading Failure**: If the model is not found or fails to load securely, `/predict` cleanly returns (`503 Service Unavailable`).

## Development Usage
To spin up the server locally on port 8000:
```bash
cd backend
uvicorn app.main:app --reload
```
