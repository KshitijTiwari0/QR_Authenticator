import cv2
import numpy as np
from skimage.util import view_as_blocks

class QRFeatureExtractor:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.feature_names = [
            'mean_intensity', 'std_intensity',
            'edge_sharpness', 'global_contrast', 'local_contrast_var',
            'finder_pattern_score', 'alignment_deviation',
            'high_freq_ratio', 'spectral_std',
            'ink_area_variance', 'ink_eccentricity'
        ]
        
    def extract(self, image):
        """Main feature extraction method
        Args:
            image: Input grayscale QR code image
        Returns:
            np.array: Concatenated feature vector
        """
        image = cv2.resize(image, self.img_size)
        return np.concatenate([
            self._basic_properties(image),
            self._print_quality_features(image),
            self._structural_analysis(image),
            self._spectral_analysis(image),
            self._ink_analysis(image)
        ])
    
    def _basic_properties(self, image):
        """Extract basic image statistics"""
        return [np.mean(image), np.std(image)]
    
    def _print_quality_features(self, image):
        """Extract print quality metrics"""
        # Edge sharpness
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        edge_sharpness = laplacian.var()
        
        # Global contrast
        contrast = np.max(image) - np.min(image)
        
        # Local contrast variance
        local_contrast = self._local_contrast_variance(image)
        
        return [edge_sharpness, contrast, local_contrast]
    
    def _local_contrast_variance(self, image):
        """Calculate variance of local contrast in image blocks"""
        block_size = (32, 32)
        if image.shape[0] < block_size[0] or image.shape[1] < block_size[1]:
            return 0
        
        windows = view_as_blocks(image, block_size)
        local_contrasts = [w.max() - w.min() for w in windows.reshape(-1, *block_size)]
        return np.var(local_contrasts)
    
    def _structural_analysis(self, image):
        """Analyze QR code structural patterns"""
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Finder pattern consistency (top-left corner)
        finder_region = binary[:21, :21]  # Assuming version 1 QR code
        finder_score = np.mean(finder_region == 255)
        
        # Alignment pattern detection
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_boxes = [
            cv2.boundingRect(c)[:2] 
            for c in contours 
            if 50 < cv2.contourArea(c) < 150
        ]
        alignment_deviation = np.std(bounding_boxes) if bounding_boxes else 0
        
        return [finder_score, alignment_deviation]
    
    def _spectral_analysis(self, image):
        """Extract frequency domain features"""
        fft = np.fft.fft2(image)
        fshift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-9)  # Avoid log(0)
        
        # High frequency components (30-70px from center)
        high_freq = magnitude[30:70, 30:70].mean()
        
        # Low frequency components (100-150px from center)
        low_freq = (
            magnitude[100:150, 100:150].mean() 
            if magnitude.shape[0] > 150 
            else 0
        )
        
        return [
            high_freq / (low_freq + 1e-9),  # High-to-low frequency ratio
            np.std(magnitude)                # Spectral standard deviation
        ]
    
    def _ink_analysis(self, image):
        """Analyze ink distribution patterns"""
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Ink blob areas
        areas = [cv2.contourArea(c) for c in contours]
        
        # Ink blob eccentricities
        eccentricities = []
        for c in contours:
            if len(c) >= 5:  # Minimum points needed for ellipse fitting
                _, (major, minor), _ = cv2.fitEllipse(c)
                if major > 0:
                    eccentricities.append(np.sqrt(1 - (minor / major) ** 2))
        
        return [
            np.var(areas) if areas else 0,          # Area variance
            np.mean(eccentricities) if eccentricities else 0  # Mean eccentricity
        ]