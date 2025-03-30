import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GLCMFeatureExtractor:
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles
        self.props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
    def extract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        glcm = graycomatrix(gray, distances=self.distances, angles=self.angles,
                           levels=256, symmetric=True, normed=True)
        return np.array([graycoprops(glcm, prop).mean() for prop in self.props])
