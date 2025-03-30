import os
import cv2
import numpy as np
from pathlib import Path

class HybridDataLoader:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        
    def load_images(self, folder_path, label):
        images = []
        labels = []
        for img_path in Path(folder_path).glob('*'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(label)
        return images, np.array(labels)
    
    def load_dataset(self, original_path, counterfeit_path):
        orig_imgs, orig_labels = self.load_images(original_path, 0)
        cntft_imgs, cntft_labels = self.load_images(counterfeit_path, 1)
        return orig_imgs + cntft_imgs, np.concatenate([orig_labels, cntft_labels])
