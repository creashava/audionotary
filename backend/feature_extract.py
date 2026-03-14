import os
import glob
import numpy as np
import pandas as pd
import librosa

TARGET_SR = 16000


def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    y = librosa.util.normalize(y)
    return y, sr


def extract_features(filepath):

    y, sr = load_audio(filepath)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {}

    for i in range(20):
        features[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i}_std"] = float(np.std(mfcc[i]))

    features["rms"] = float(np.mean(librosa.feature.rms(y=y)))
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return features


def build_dataset(real_dir, fake_dir, output_csv="features.csv"):

    rows = []

    for label, folder in [("real", real_dir), ("fake", fake_dir)]:

        files = glob.glob(os.path.join(folder, "*.wav"))

        print("Processing", label, ":", len(files), "files")

        for fp in files:

            try:
                feat = extract_features(fp)

                feat["label"] = 1 if label == "fake" else 0
                feat["filepath"] = fp

                rows.append(feat)

            except Exception as e:
                print("Skipped", fp, e)

    df = pd.DataFrame(rows)

    df.to_csv(output_csv, index=False)

    print("Dataset saved:", output_csv)

    return df