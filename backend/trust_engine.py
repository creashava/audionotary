def compute_trust_score(l1_result, l2_result):

    l1 = l1_result["l1_score"]
    l2 = l2_result["l2_score"]

    trust_score = (l1 * 0.4) + (l2 * 0.6)

    if trust_score > 70:
        verdict = "AUTHENTIC"
    else:
        verdict = "SYNTHETIC"

    return {
        "trust_score": trust_score,
        "verdict": verdict,
        "l1_score": l1,
        "l2_score": l2
    }