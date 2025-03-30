from data.data_loader import QRDataLoader
from features.feature_extractor import QRFeatureExtractor
from models.train import QRModelTrainer
from config.paths import DATA_PATHS

def main():
    # Load data
    loader = QRDataLoader()
    X, y = loader.load_dataset(DATA_PATHS['original'], DATA_PATHS['counterfeit'])
    
    # Extract features
    extractor = QRFeatureExtractor()
    X_features = np.array([extractor.extract(img) for img in X])
    
    # Train model
    trainer = QRModelTrainer()
    trainer.train(X_features, y)
    trainer.save_model()
    
    # Feature importance
    plot_feature_importance(extractor.feature_names, trainer.model.feature_importances_)

if __name__ == "__main__":
    main()
