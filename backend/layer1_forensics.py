import os

def analyze_layer1(filepath):

    file_size = os.path.getsize(filepath)

    flags = []
    score = 100

    if file_size < 1000:
        flags.append("File too small")
        score -= 20

    return {
        "l1_score": score,
        "flags": flags
    }