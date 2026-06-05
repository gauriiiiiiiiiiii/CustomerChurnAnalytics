# config.py
# Centralizes the path to the trained model artifact.
# BASE_DIR resolves to the project root (one level up from src/).
# MODEL_PATH points to models/best_model.joblib — the saved sklearn Pipeline
# that contains both the preprocessor (encoding + scaling) and the trained classifier.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
