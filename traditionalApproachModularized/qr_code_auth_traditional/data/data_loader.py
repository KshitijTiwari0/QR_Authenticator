import os
import cv2
import numpy as np
from pathlib import Path

class QRDataLoader:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        
    def load_from_folder(self, folder_path, label):
        """Load images from a folder with resizing"""
        images, labels = [], []
        for filename in os.listdir(folder_path):
            img_path = Path(folder_path) / filename
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(label)
        return images, labels

    def load_dataset(self, original_path, counterfeit_path):
        """Load complete dataset"""
        orig_images, orig_labels = self.load_from_folder(original_path, 0)
        fake_images, fake_labels = self.load_from_folder(counterfeit_path, 1)
        return orig_images + fake_images, np.array(orig_labels + fake_labels)
