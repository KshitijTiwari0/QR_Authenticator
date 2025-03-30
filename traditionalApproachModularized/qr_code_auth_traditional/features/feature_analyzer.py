import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler

class QRFeatureAnalyzer:
    def __init__(self, feature_extractor):
        """
        Initialize with a feature extractor instance
        
        Args:
            feature_extractor: QRFeatureExtractor instance
        """
        self.extractor = feature_extractor
        self.feature_names = feature_extractor.feature_names
        
    def analyze_dataset(self, original_folder: str, counterfeit_folder: str) -> Tuple[Dict, np.ndarray]:
        """
        Analyze complete dataset and extract features
        
        Args:
            original_folder: Path to original QR codes
            counterfeit_folder: Path to counterfeit QR codes
            
        Returns:
            Tuple containing:
                - Dictionary of features (key: feature name, value: array of values)
                - Array of labels (0 for original, 1 for counterfeit)
        """
        # Load images
        orig_imgs, orig_labels = self._load_images(original_folder, 0)
        counterfeit_imgs, counterfeit_labels = self._load_images(counterfeit_folder, 1)
        
        # Combine datasets
        all_imgs = orig_imgs + counterfeit_imgs
        all_labels = np.concatenate([orig_labels, counterfeit_labels])
        
        # Extract features
        features = {name: [] for name in self.feature_names}
        for img in all_imgs:
            feature_vector = self.extractor.extract(img)
            for i, name in enumerate(self.feature_names):
                features[name].append(feature_vector[i])
                
        # Convert to numpy arrays
        features = {k: np.array(v) for k, v in features.items()}
        
        return features, all_labels
    
    def _load_images(self, folder_path: str, label: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load images from folder with resizing
        
        Args:
            folder_path: Path to image folder
            label: Class label for these images
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        for img_path in Path(folder_path).glob('*'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
        return images, np.array(labels)
    
    def get_feature_stats(self, features: Dict, labels: np.ndarray) -> pd.DataFrame:
        """
        Calculate descriptive statistics for features by class
        
        Args:
            features: Dictionary of features
            labels: Array of labels
            
        Returns:
            DataFrame with statistics for each feature
        """
        stats = []
        for name, values in features.items():
            orig_stats = {
                'feature': name,
                'class': 'original',
                'mean': np.mean(values[labels == 0]),
                'std': np.std(values[labels == 0]),
                'min': np.min(values[labels == 0]),
                'max': np.max(values[labels == 0])
            }
            counterfeit_stats = {
                'feature': name,
                'class': 'counterfeit',
                'mean': np.mean(values[labels == 1]),
                'std': np.std(values[labels == 1]),
                'min': np.min(values[labels == 1]),
                'max': np.max(values[labels == 1])
            }
            stats.extend([orig_stats, counterfeit_stats])
            
        return pd.DataFrame(stats)
    
    def plot_feature_distributions(self, features: Dict, labels: np.ndarray, 
                                 feature_names: List[str] = None, 
                                 cols: int = 3, figsize: Tuple = (15, 10)):
        """
        Plot distributions of features comparing original vs counterfeit
        
        Args:
            features: Dictionary of features
            labels: Array of labels
            feature_names: List of features to plot (None for all)
            cols: Number of columns in subplot grid
            figsize: Figure size
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        rows = int(np.ceil(len(feature_names) / cols))
        plt.figure(figsize=figsize)
        
        for i, name in enumerate(feature_names, 1):
            plt.subplot(rows, cols, i)
            sns.histplot(features[name][labels == 0], 
                         color='blue', label='Original', kde=True)
            sns.histplot(features[name][labels == 1], 
                         color='red', label='Counterfeit', kde=True)
            plt.title(name)
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlations(self, features: Dict, labels: np.ndarray,
                                figsize: Tuple = (12, 10)):
        """
        Plot correlation matrix of features
        
        Args:
            features: Dictionary of features
            labels: Array of labels
            figsize: Figure size
        """
        # Create DataFrame
        df = pd.DataFrame(features)
        df['label'] = labels
        
        # Compute correlations
        corr = df.corr()
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()
    
    def plot_normalized_feature_ranges(self, features: Dict, labels: np.ndarray,
                                      figsize: Tuple = (10, 6)):
        """
        Plot normalized feature ranges by class
        
        Args:
            features: Dictionary of features
            labels: Array of labels
            figsize: Figure size
        """
        # Normalize features
        scaler = StandardScaler()
        normalized = scaler.fit_transform(pd.DataFrame(features))
        normalized = pd.DataFrame(normalized, columns=self.feature_names)
        
        # Calculate mean ± std for each feature by class
        orig_means = normalized[labels == 0].mean()
        orig_stds = normalized[labels == 0].std()
        counterfeit_means = normalized[labels == 1].mean()
        counterfeit_stds = normalized[labels == 1].std()
        
        # Create plot
        plt.figure(figsize=figsize)
        x = np.arange(len(self.feature_names))
        width = 0.35
        
        plt.bar(x - width/2, orig_means, width, yerr=orig_stds,
               label='Original', color='blue', alpha=0.7)
        plt.bar(x + width/2, counterfeit_means, width, yerr=counterfeit_stds,
               label='Counterfeit', color='red', alpha=0.7)
        
        plt.xticks(x, self.feature_names, rotation=45, ha='right')
        plt.ylabel('Normalized Feature Value')
        plt.title('Normalized Feature Ranges by Class')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, importances: np.ndarray, 
                              figsize: Tuple = (10, 6)):
        """
        Plot feature importance scores
        
        Args:
            importances: Array of importance scores
            figsize: Figure size
        """
        # Sort features by importance
        sorted_idx = np.argsort(importances)
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(sorted_names)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Feature Importance Score')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()
