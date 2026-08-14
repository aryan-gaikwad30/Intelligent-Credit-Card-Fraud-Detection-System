# Phase 10: End-to-End Integration & Production Hardening

This document outlines the final integration architecture, E2E testing framework, and production hardening applied to the Intelligent Credit Card Fraud Detection System.

## 1. System Architecture
The application runs as a decoupled system:
- **Frontend**: React SPA served via Vite (typically on port 5173).
- **Backend**: FastAPI ML inference service (typically on port 8000).
- **Communication**: Strict REST JSON payloads. The frontend enforces basic numeric bounds before transmitting the 30-feature vector to the backend.
- **Model Lifecycle**: The backend securely loads the locked `Phase 4 XGBoost Baseline` model exactly once during startup (`lifespan`). The `final_model_config.json` enforces the exact threshold (`0.31`) and feature expectations.

## 2. Environment Configuration
The project securely separates configurations:
- **Frontend** (`frontend/.env.example`): Exposes `VITE_API_BASE_URL` which the Vite client intercepts and maps to API calls in `client.js`.
- **Backend** (OS Environment): Reads `BACKEND_CORS_ORIGINS` natively via `os.environ` to dynamically configure CORS in `main.py`, avoiding insecure `["*"]` wildcards in production. No `.env` parser is strictly required unless deployed to a container orchestrator.

## 3. Server Startup
**Backend**:
Run from the root directory:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
This binds the FastAPI server and immediately validates model artifacts.

**Frontend**:
Run from the `frontend` directory:
```bash
npm run dev
```
Alternatively, build for production using `npm run build`.

## 4. API Contract & Error Handling
The backend exclusively exposes:
- `GET /api/v1/health`: Returns API readiness and model load status.
- `POST /api/v1/predict`: Accepts the raw 30-feature vector and returns the `fraud_probability`, `threshold`, and `is_fraud` flags.

**Hardening Rules**:
- **NaN/Infinity**: Pydantic strictly rejects non-finite numerics before hitting the ML pipeline.
- **Exceptions**: A global exception handler (`@app.exception_handler(Exception)`) catches all unhandled internal ML errors, logs the full stack trace securely to the server logs, and returns a safe HTTP 500 without leaking Python internal paths to the client.

## 5. End-to-End Verification

The integration verification uses a robust Python-to-Browser E2E lifecycle:
1. `backend/tests/test_e2e_integration.py` binds a free socket port and launches `uvicorn` in a subprocess.
2. The Python HTTP client verifies the health endpoint and runs a raw HTTP prediction using a deterministic `fixtures/synthetic_transaction.json`.
3. The script then dynamically passes the backend URL to `npx playwright test` and executes a genuine Chromium browser test via `frontend/e2e/dashboard.spec.js`.
4. The Playwright script navigates the Vite UI, fills out the form, submits the transaction, and asserts the UI renders the correct 31.00% threshold and XGBoost identity without hard-coding assumptions.
5. Finally, the subprocess is gracefully terminated.

To run the full suite:
```bash
pytest backend/tests/
```

## 6. Known Limitations
- The system is "production-oriented" but lacks production identity/authentication layers (OAuth/JWT) which may be required prior to public internet exposure.
- Rate limiting is not currently implemented on the `/predict` endpoint.
