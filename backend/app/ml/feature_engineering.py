import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.seconds_per_day = 86400
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_eng = X.copy()
        
        if 'Time' in X_eng.columns:
            # Deterministic Time transformations
            X_eng['time_seconds_in_day'] = X_eng['Time'] % self.seconds_per_day
            X_eng['time_of_day_sin'] = np.sin(X_eng['time_seconds_in_day'] * (2 * np.pi / self.seconds_per_day))
            X_eng['time_of_day_cos'] = np.cos(X_eng['time_seconds_in_day'] * (2 * np.pi / self.seconds_per_day))
            X_eng = X_eng.drop(columns=['Time', 'time_seconds_in_day'])
            
        if 'Amount' in X_eng.columns:
            # Deterministic Amount transformation
            X_eng['amount_log1p'] = np.log1p(X_eng['Amount'])
            X_eng = X_eng.drop(columns=['Amount'])
            
        return X_eng
