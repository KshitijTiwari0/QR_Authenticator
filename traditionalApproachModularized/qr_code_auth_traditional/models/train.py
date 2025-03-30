import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from config.paths import MODEL_PATHS

class QRModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=random_state
        )
        
    def train(self, X, y):
        """Full training pipeline"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Preprocessing
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Training
        self.model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        return self
    
    def save_model(self):
        """Save model and scaler"""
        joblib.dump(self.model, MODEL_PATHS['model'])
        joblib.dump(self.scaler, MODEL_PATHS['scaler'])
