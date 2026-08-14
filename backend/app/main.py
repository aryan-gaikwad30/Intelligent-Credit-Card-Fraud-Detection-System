from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import logging
import math

from backend.app.core.config import config
from backend.app.api.endpoints import router as api_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API...")
    try:
        # Load model securely based on config rules
        logger.info(f"Loading model artifact from {config.model_artifact}...")
        model = joblib.load(config.model_artifact)
        
        # Bind to app state
        app.state.model = model
        app.state.threshold = config.threshold
        app.state.feature_columns = config.feature_columns
        app.state.model_name = config.model_name
        logger.info(f"Successfully loaded {config.model_name} with threshold {config.threshold}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Allow startup but predict will fail safely
        app.state.model = None
    
    yield
    # Shutdown
    logger.info("Shutting down API...")
    app.state.model = None

app = FastAPI(
    title="Intelligent Credit Card Fraud Detection API",
    description="Inference API for real-time fraud detection.",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Strip the "input" key from the detail to avoid JSON serialization issues with NaN/Inf
    errors = exc.errors()
    for error in errors:
        if "input" in error:
            # We can't safely serialize NaN/Inf, so just remove the input context
            del error["input"]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud Detection API is running. Check /api/v1/health"}
