# Security and Hardening

## Overview
This document outlines the security controls, validations, and hardening measures implemented for the Intelligent Credit Card Fraud Detection System.

## Architecture and Access
* **Frontend/Backend Separation**: The frontend React app operates independently from the FastAPI backend, connected securely via API endpoints.
* **CORS Restrictions**: The backend uses environment-driven CORS configuration (`BACKEND_CORS_ORIGINS`). This prevents wildcard `*` access in production and only allows verified frontend domains (like Vercel) to interact with the service.

## Data and API Validation
* **Strict Input Validation**: The `/predict` endpoint employs Pydantic's `ConfigDict(strict=True)` along with `StrictNumeric` checks on all 30 feature fields. 
* **NaN/Infinity Rejection**: The API explicitly refuses `NaN` and `Infinity` using `allow_inf_nan=False` in its data models. Malformed payloads or incorrect types are rejected before reaching the ML model inference code.
* **Error Handling**: The `validation_exception_handler` safely strips out context to avoid serialization issues, while the `global_exception_handler` catches any unexpected backend failure and returns a generic HTTP 500 without exposing sensitive server-side exceptions, stack traces, or model internals to the client.

## Model Security
* **Model Integrity Verification**: During startup, `backend/app/core/config.py` computes the SHA-256 hash of `models/xgboost_baseline.joblib` and verifies it against the known-good hash in `models/final_model_config.json`. If the hashes mismatch, the system safely fails to start, preventing tampering or corrupted model loading.

## Secrets Management
* **Environment Variables**: Sensitive configuration (like allowed CORS origins or API bases) is provided via environment variables, not hardcoded into the source. No `.env` files are checked into version control.

## Logging and Monitoring
* **Server-Side Logging**: The backend records critical startup phases, model integrity checks, and exceptions securely to server logs. The logs avoid emitting full request vectors or client PII. 

## Dependency and Build Security
* **Dependency Auditing**: Regular dependency audits run to ensure package versions are safe and minimal, keeping the attack surface small. (e.g., using `npm audit`).

## Automated Testing
* **Test Matrix**: The system includes a robust suite of over 50 unit and integration tests across backend and frontend, validating the numerical validation rules, edge cases, and e2e integration flow.

## Security Limitations
* **Rate Limiting**: The current deployment on Render does not enforce application-layer rate limiting (such as `slowapi` or Redis-backed quotas). A dedicated WAF or infrastructure rate limiting would be needed for enterprise-grade flood protection.
