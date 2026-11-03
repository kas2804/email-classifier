# src/api.py
import os
# pyrefly: ignore[missing-import]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load

try:
    from src.clean import clean_email
except ImportError:
    try:
        from .clean import clean_email
    except ImportError:
        from clean import clean_email

app = FastAPI(title="Email Classifier API")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classifier.joblib")

_bundle = None

def get_model_bundle(model_path: str = DEFAULT_MODEL_PATH):
    """
    Lazily loads the trained model bundle into memory on first request.
    """
    global _bundle
    if _bundle is None:
        target_path = model_path
        if not os.path.exists(target_path):
            if os.path.exists("models/classifier.joblib"):
                target_path = "models/classifier.joblib"
            else:
                raise FileNotFoundError(f"Model file not found at {model_path} or models/classifier.joblib")
        _bundle = load(target_path)
    return _bundle

class EmailIn(BaseModel):
    subject: str | None = ""
    body: str | None = ""

@app.post("/predict")
def predict(x: EmailIn):
    try:
        bundle = get_model_bundle()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")

    model = bundle["model"]
    labels = bundle["labels"]

    # Combine subject and body for classification
    text = f"{x.subject or ''} {x.body or ''}".strip()
    cleaned_text = clean_email(text)

    # Get single best prediction and probabilities for all classes
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]

    # Map probabilities to their labels
    proba_map = {label: float(prob) for label, prob in zip(labels, probabilities)}

    return {
        "prediction": prediction,
        "probabilities": proba_map,
    }

if __name__ == "__main__":
    try:
        # pyrefly: ignore[missing-import]
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Error: 'uvicorn' is not installed in the currently active Python environment.")
        print("Activate your project virtual environment first:\n  source .venv/bin/activate\nOr install requirements:\n  pip install -r requirements.txt")
