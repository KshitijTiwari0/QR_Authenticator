import joblib
import cv2
import numpy as np
from config.paths import MODEL_PATHS

class QRAuthenticator:
    def __init__(self):
        self.model = joblib.load(MODEL_PATHS['model'])
        self.scaler = joblib.load(MODEL_PATHS['scaler'])
        self.feature_extractor = QRFeatureExtractor()
        
    def authenticate(self, image_path):
        """Complete authentication pipeline"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = self.feature_extractor.extract(img)
        features = self.scaler.transform([features])
        proba = self.model.predict_proba(features)[0][1]
        return {
            'authentic': proba < 0.5,
            'confidence': 1 - abs(proba - 0.5) * 2
        }
