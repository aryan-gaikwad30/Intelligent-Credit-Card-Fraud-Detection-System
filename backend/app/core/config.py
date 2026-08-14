import json
import os

class AppConfig:
    def __init__(self, config_path="models/final_model_config.json"):
        if not os.path.exists(config_path):
            raise RuntimeError(f"Configuration file {config_path} not found.")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.model_name = config.get("model_name")
        self.threshold = config.get("threshold")
        self.model_artifact = config.get("model_artifact")
        self.feature_columns = config.get("feature_columns")
        
        # Read CORS from OS environment
        cors_env = os.environ.get("BACKEND_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        self.cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        
        self._validate()
        
    def _validate(self):
        # Validate against the locked Phase 4 XGBoost baseline requirements
        if self.model_name != "Phase 4 XGBoost Baseline":
            raise ValueError(f"Invalid model_name in config: expected 'Phase 4 XGBoost Baseline', got '{self.model_name}'")
        
        if abs(self.threshold - 0.31) > 1e-6:
            raise ValueError(f"Invalid threshold in config: expected 0.31, got {self.threshold}")
            
        if self.model_artifact != "models/xgboost_baseline.joblib":
            raise ValueError(f"Invalid model_artifact in config: expected 'models/xgboost_baseline.joblib', got '{self.model_artifact}'")
            
        if not os.path.exists(self.model_artifact):
            raise RuntimeError(f"Model artifact {self.model_artifact} not found.")
            
        if not self.feature_columns or not isinstance(self.feature_columns, list):
            raise ValueError("Invalid or missing feature_columns in config.")

config = AppConfig()
