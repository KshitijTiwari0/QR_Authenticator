import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Union
from pathlib import Path
import joblib

class QRCodeAuthenticator:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the QR code authenticator with trained model and scaler
        
        Args:
            model_path: Path to saved TensorFlow model
            scaler_path: Path to saved GLCM feature scaler
        """
        # Load model and scaler
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Initialize feature extractors
        self.glcmextractor = GLCMFeatureExtractor()
        self.preprocessor = EfficientNetPreprocessor()
        
        # Build EfficientNet feature extractor
        self.effnet_encoder = self._build_effnet_encoder()

    def _build_effnet_encoder(self):
        """Build EfficientNetB0 feature extractor with frozen weights"""
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        encoder = tf.keras.Model(inputs=base_model.input, outputs=x)
        
        # Freeze all layers
        for layer in encoder.layers:
            layer.trainable = False
            
        return encoder

    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract both GLCM and EfficientNet features from an image
        
        Args:
            image: Input BGR image (numpy array)
            
        Returns:
            Dictionary containing:
                - 'glcm': GLCM features (6-dim)
                - 'effnet': EfficientNet embeddings (1280-dim)
        """
        # Extract GLCM features
        glcm_features = self.glcmextractor.extract(image)
        
        # Extract EfficientNet features
        preprocessed = self.preprocessor.preprocess(image)
        effnet_features = self.effnet_encoder.predict(
            np.expand_dims(preprocessed, axis=0))[0]
        
        return {
            'glcm': glcm_features,
            'effnet': effnet_features
        }

    def authenticate(self, image_path: Union[str, Path], 
                    threshold: float = 0.5) -> Dict[str, Union[bool, float]]:
        """
        Authenticate a QR code image
        
        Args:
            image_path: Path to the image file
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Dictionary with authentication results:
                - 'authentic': True if original, False if counterfeit
                - 'confidence': Confidence score (0-1)
                - 'features': Extracted feature values
        """
        # Load and validate image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Extract features
        features = self.extract_features(img)
        
        # Scale GLCM features
        scaled_glcm = self.scaler.transform([features['glcm']])
        
        # Make prediction
        proba = self.model.predict([
            np.array([features['effnet']]), 
            scaled_glcm
        ])[0][0]
        
        return {
            'authentic': proba < threshold,
            'confidence': 1 - abs(proba - threshold) * 2,  # Normalize to 0-1
            'features': {
                'glcm': features['glcm'].tolist(),
                'effnet': features['effnet'].tolist()
            }
        }

    def batch_authenticate(self, image_paths: List[Union[str, Path]], 
                         threshold: float = 0.5) -> Dict[str, list]:
        """
        Authenticate multiple QR code images
        
        Args:
            image_paths: List of image paths
            threshold: Decision threshold
            
        Returns:
            Dictionary with lists of results for each image
        """
        results = {
            'authentic': [],
            'confidence': [],
            'features': []
        }
        
        for path in image_paths:
            try:
                result = self.authenticate(path, threshold)
                results['authentic'].append(result['authentic'])
                results['confidence'].append(result['confidence'])
                results['features'].append(result['features'])
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                results['authentic'].append(None)
                results['confidence'].append(None)
                results['features'].append(None)
                
        return results

# Helper class definitions (would normally be imported from other modules)
class GLCMFeatureExtractor:
    # ... (same implementation as in glcm_extractor.py)
    pass

class EfficientNetPreprocessor:
    # ... (same implementation as in preprocessing.py)
    pass

if __name__ == "__main__":
    # Example usage
    authenticator = QRCodeAuthenticator(
        model_path="saved_models/hybrid_model.h5",
        scaler_path="saved_models/glcm_scaler.pkl"
    )
    
    # Single prediction
    result = authenticator.authenticate("test_qr.png")
    print(f"Authentic: {result['authentic']}, Confidence: {result['confidence']:.2f}")
    
    # Batch prediction
    results = authenticator.batch_authenticate(["qr1.png", "qr2.png"])
    for i, (auth, conf) in enumerate(zip(results['authentic'], results['confidence'])):
        print(f"QR {i+1}: Authentic={auth}, Confidence={conf:.2f}")
