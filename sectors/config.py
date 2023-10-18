from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

DATA_DIR = BASE_DIR / "data"
INDUSTRY_DATA_DIR = DATA_DIR / "industries"
HATESPEECH_DATA_DIR = DATA_DIR / "hatespeech"

RESULTS_DIR = BASE_DIR / "results"
INDUSTRY_RESULTS_DIR = RESULTS_DIR / "industries"
HATESPEECH_RESULTS_DIR = RESULTS_DIR / "hatespeech"

MODELS_DIR = BASE_DIR / "models"
INDUSTRY_MODELS_DIR = MODELS_DIR / "industries"
HATESPEECH_MODELS_DIR = MODELS_DIR / "hatespeech"

FIG_DIR = BASE_DIR / "figures"

TENSORBOARD_DIR = BASE_DIR / "runs"
