from collections import Counter


ENV_KEYWORDS = {
    "carbon", "emissions", "co2", "energy", "water", "recycling",
    "renewable", "waste", "climate", "pollution", "scope 1", "scope 2"
}

SOCIAL_KEYWORDS = {
    "employee", "workers", "safety", "injury", "diversity", "women",
    "human rights", "training", "labor", "strike", "turnover", "wellbeing"
}

GOV_KEYWORDS = {
    "board", "governance", "bribery", "audit", "shareholder",
    "transparency", "compliance", "ethics", "oversight", "independence"
}


def extract_keywords(text: str, keywords: set[str]) -> list[str]:
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return sorted(found)


def simple_sentiment(text: str) -> str:
    positive_words = {"improved", "strong", "growth", "reduced", "positive", "enhanced"}
    negative_words = {"risk", "high", "incident", "decline", "violation", "bribery", "strike"}

    words = text.lower().split()
    counts = Counter(words)

    pos = sum(counts[w] for w in positive_words if w in counts)
    neg = sum(counts[w] for w in negative_words if w in counts)

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def analyze_esg_text(text: str) -> dict:
    env_hits = extract_keywords(text, ENV_KEYWORDS)
    social_hits = extract_keywords(text, SOCIAL_KEYWORDS)
    gov_hits = extract_keywords(text, GOV_KEYWORDS)

    alerts = []
    lower_text = text.lower()

    if "high co2" in lower_text or "high emissions" in lower_text:
        alerts.append("High emissions risk mentioned.")
    if "bribery" in lower_text:
        alerts.append("Governance risk: bribery mentioned.")
    if "strike" in lower_text or "labor dispute" in lower_text:
        alerts.append("Social risk: labor conflict mentioned.")

    return {
        "environmental_keywords": env_hits,
        "social_keywords": social_hits,
        "governance_keywords": gov_hits,
        "sentiment": simple_sentiment(text),
        "alerts": alerts,
    }