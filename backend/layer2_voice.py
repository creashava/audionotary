import numpy as np
import joblib
from feature_extract import extract_features

# load trained model
model = joblib.load("../audio_model.pkl")

def analyze_layer2(filepath):

    # extract features
    features = extract_features(filepath)

    # convert to numpy
    X = np.array(list(features.values())).reshape(1, -1)

    # predict
    prediction = model.predict(X)[0]

    if prediction == 1:
        score = 20
        flags = ["Synthetic voice detected"]
    else:
        score = 85
        flags = []

    return {
        "l2_score": score,
        "bio_features": features,
        "bio_flags": flags
    }