import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

class EfficientNetPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess(self, image):
        resized = cv2.resize(image, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return preprocess_input(rgb.astype(np.float32))
