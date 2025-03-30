from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    'original': str(BASE_DIR / ''),
    'counterfeit': str(BASE_DIR / '')
}

MODEL_PATHS = {
    'model': str(BASE_DIR / 'models/saved_models/random_forest.pkl'),
    'scaler': str(BASE_DIR / 'models/saved_models/scaler.pkl')
} 
