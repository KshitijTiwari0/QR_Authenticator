from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PATHS = {
    'original': '',
    'counterfeit': '',
    'model': BASE_DIR / 'saved_models/hybrid_model.h5',
    'scaler': BASE_DIR / 'saved_models/glcm_scaler.pkl'
}
