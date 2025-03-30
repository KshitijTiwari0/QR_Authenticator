import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import cv2

class QRVisualizer:
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize visualizer with optional feature names
        
        Args:
            feature_names: List of feature names for labeling
        """
        self.feature_names = feature_names or []
        plt.style.use('seaborn')
        sns.set_palette("colorblind")

    def plot_feature_distributions(self, features: Dict[str, np.ndarray], 
                                 labels: np.ndarray,
                                 feature_subset: Optional[List[str]] = None,
                                 figsize: tuple = (15, 10),
                                 cols: int = 3):
        """
        Plot distributions of features comparing original vs counterfeit
        
        Args:
            features: Dictionary of feature arrays
            labels: Array of labels (0=original, 1=counterfeit)
            feature_subset: Specific features to plot (None for all)
            figsize: Figure size
            cols: Number of columns in subplot grid
        """
        if not feature_subset:
            feature_subset = list(features.keys())
            
        rows = int(np.ceil(len(feature_subset) / cols))
        plt.figure(figsize=figsize)
        
        for i, feat in enumerate(feature_subset, 1):
            plt.subplot(rows, cols, i)
            
            # Plot original samples
            sns.kdeplot(features[feat][labels == 0], 
                       label='Original', fill=True, alpha=0.5)
            
            # Plot counterfeit samples
            sns.kdeplot(features[feat][labels == 1], 
                       label='Counterfeit', fill=True, alpha=0.5)
            
            plt.title(feat if not self.feature_names else self.feature_names[i-1])
            plt.xlabel('Feature Value')
            plt.ylabel('Density')
            plt.legend()
            
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            classes: List[str] = ['Original', 'Counterfeit'],
                            figsize: tuple = (6, 6)):
        """
        Plot a confusion matrix with annotations
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class names
            figsize: Figure size
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_scores: np.ndarray,
                      figsize: tuple = (8, 6)):
        """
        Plot ROC curve with AUC score
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            figsize: Figure size
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_training_history(self, history: Dict[str, List[float]],
                            metrics: List[str] = ['loss', 'accuracy'],
                            figsize: tuple = (12, 5)):
        """
        Plot training history curves
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, len(metrics), i)
            
            # Plot training metric
            plt.plot(history[metric], label=f'Training {metric}')
            
            # Plot validation metric if exists
            val_metric = f'val_{metric}'
            if val_metric in history:
                plt.plot(history[val_metric], label=f'Validation {metric}')
            
            plt.title(f'Training and Validation {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, importances: np.ndarray,
                              figsize: tuple = (10, 6)):
        """
        Plot horizontal bar chart of feature importances
        
        Args:
            importances: Array of feature importance scores
            figsize: Figure size
        """
        if not self.feature_names:
            raise ValueError("Feature names not provided during initialization")
            
        sorted_idx = np.argsort(importances)
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(sorted_names)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

    def display_qr_samples(self, image_paths: List[str],
                         labels: Optional[List[str]] = None,
                         cols: int = 5,
                         figsize: tuple = (15, 4)):
        """
        Display a grid of QR code samples
        
        Args:
            image_paths: List of image paths
            labels: Optional list of labels for each image
            cols: Number of columns in grid
            figsize: Figure size
        """
        rows = int(np.ceil(len(image_paths) / cols))
        plt.figure(figsize=figsize)
        
        for i, img_path in enumerate(image_paths, 1):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            plt.subplot(rows, cols, i)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            if labels and i <= len(labels):
                plt.title(labels[i-1])
        
        plt.tight_layout()
        plt.show()

    def plot_feature_correlations(self, features: Dict[str, np.ndarray],
                                figsize: tuple = (10, 8)):
        """
        Plot correlation matrix of features
        
        Args:
            features: Dictionary of feature arrays
            figsize: Figure size
        """
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        if self.feature_names:
            df.columns = self.feature_names
            
        # Compute correlations
        corr = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_threshold_analysis(self, y_true: np.ndarray,
                              y_scores: np.ndarray,
                              metrics: List[str] = ['f1', 'precision', 'recall'],
                              figsize: tuple = (10, 6)):
        """
        Plot metric curves across different decision thresholds
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            metrics: Metrics to evaluate
            figsize: Figure size
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        thresholds = np.linspace(0, 1, 50)
        metric_values = {metric: [] for metric in metrics}
        
        for thresh in thresholds:
            y_pred = (y_scores > thresh).astype(int)
            if 'f1' in metrics:
                metric_values['f1'].append(f1_score(y_true, y_pred))
            if 'precision' in metrics:
                metric_values['precision'].append(precision_score(y_true, y_pred))
            if 'recall' in metrics:
                metric_values['recall'].append(recall_score(y_true, y_pred))
        
        plt.figure(figsize=figsize)
        for metric in metrics:
            plt.plot(thresholds, metric_values[metric], label=metric.capitalize())
        
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Metric Scores vs. Decision Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
