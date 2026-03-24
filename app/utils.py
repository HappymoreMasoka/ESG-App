def risk_label(score: float) -> str:
    if score >= 75:
        return "Low Risk"
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


def score_color(score: float) -> str:
    if score >= 75:
        return "green"
    if score >= 50:
        return "orange"
    return "red"