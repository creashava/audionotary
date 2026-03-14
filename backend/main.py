from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import joblib
import pandas as pd

from utils.feature_extract import extract_features

app = FastAPI(title="AudioNotary API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("models/audio_model.pkl")


@app.get("/")
def home():
    return {"message": "AudioNotary backend running"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:

        # Extract audio features
        features = extract_features(temp_path)

        df = pd.DataFrame([features])

        # Model prediction
        prediction = model.predict(df)[0]

        # Probability if available
        try:
            prob = model.predict_proba(df)[0]
            trust_score = float(max(prob) * 100)
        except:
            trust_score = 80.0

        # Flags
        flags = []

        if prediction == 1:
            flags.append("AI Generated Voice Detected")
        else:
            flags.append("Voice appears authentic")

        result = {
            "prediction": int(prediction),
            "trust_score": round(trust_score, 2),
            "flags": flags
        }

        return result

    finally:
        # Clean temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)