from config.paths import PATHS
from data.data_loader import HybridDataLoader
from features.glcm_extractor import GLCMFeatureExtractor
from models.architecture import HybridModelBuilder
from models.trainer import HybridModelTrainer
from utils.preprocessing import EfficientNetPreprocessor

def main():
    # Initialize components
    loader = HybridDataLoader()
    glcm_extractor = GLCMFeatureExtractor()
    effnet_preprocessor = EfficientNetPreprocessor()
    model_builder = HybridModelBuilder()
    trainer = HybridModelTrainer()
    
    # Load data
    images, labels = loader.load_dataset(PATHS['original'], PATHS['counterfeit'])
    
    # Extract features
    print("Extracting GLCM features...")
    glcm_features = np.array([glcm_extractor.extract(img) for img in images])
    
    # Extract EfficientNet features
    print("Extracting EfficientNet embeddings...")
    encoder = model_builder.build_encoder()
    effnet_features = np.array([encoder.predict(effnet_preprocessor.preprocess(img))[0] for img in images])
    
    # Train model
    splits = trainer.train(effnet_features, glcm_features, labels)
    model = model_builder.build_hybrid(effnet_features.shape[1], glcm_features.shape[1])
    
    history = model.fit(
        [splits['train'][0], splits['train'][1]],
        splits['train'][2],
        validation_data=([splits['test'][0], splits['test'][1]], splits['test'][2]),
        epochs=30,
        batch_size=8,
        callbacks=trainer.get_callbacks()
    )
    
    # Evaluate
    trainer.evaluate(model, splits['test'][0], splits['test'][1], splits['test'][2])
    
    # Save model
    model.save(PATHS['model'])

if __name__ == "__main__":
    main()
