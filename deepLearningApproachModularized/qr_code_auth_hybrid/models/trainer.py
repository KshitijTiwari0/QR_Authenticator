import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class HybridModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def train(self, effnet_features, glcm_features, labels):
        # Split dataset
        splits = train_test_split(glcm_features, effnet_features, labels, 
                                test_size=self.test_size, 
                                stratify=labels, 
                                random_state=self.random_state)
        
        # Scale GLCM features
        X_train_glcm = self.scaler.fit_transform(splits[0])
        X_test_glcm = self.scaler.transform(splits[1])
        
        return {
            'train': (splits[2], X_train_glcm, splits[4]),
            'test': (splits[3], X_test_glcm, splits[5])
        }
    
    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    
    def evaluate(self, model, X_test_eff, X_test_glcm, y_test):
        y_pred = (model.predict([X_test_eff, X_test_glcm]) > 0.5).astype(int)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
    def find_optimal_threshold(self, y_true, y_pred_prob):
        best_t = 0.5
        best_f1 = 0
        for t in np.linspace(0.1, 0.9, 50):
            current_f1 = f1_score(y_true, (y_pred_prob > t).astype(int))
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_t = t
        return best_t, best_f1
