"""Central configuration: file paths and tunable thresholds."""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

INTENTS_PATH = DATA_DIR / "intents.json"
UNIVERSITY_PATH = DATA_DIR / "university.json"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = MODEL_DIR / "classification_report.txt"

# Predictions below this probability fall back to a "didn't understand" reply.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10000"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
