from collections import Counter
from telemetry import tracer, nlp_counter

ENV_KEYWORDS = {
    "carbon", "emissions", "co2", "energy", "water", "recycling",
    "renewable", "waste", "climate", "pollution"
}

SOCIAL_KEYWORDS = {
    "employee", "workers", "safety", "injury", "diversity", "women",
    "human rights", "training", "labor", "strike", "turnover"
}

GOV_KEYWORDS = {
    "board", "governance", "bribery", "audit", "shareholder",
    "transparency", "compliance", "ethics", "oversight"
}


def extract_keywords(text: str, keywords: set[str]) -> list[str]:
    text_lower = text.lower()
    return sorted([kw for kw in keywords if kw in text_lower])


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
    with tracer.start_as_current_span("analyze_esg_text") as span:
        nlp_counter.add(1)

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

        span.set_attribute("nlp.environmental_hits", len(env_hits))
        span.set_attribute("nlp.social_hits", len(social_hits))
        span.set_attribute("nlp.governance_hits", len(gov_hits))
        span.set_attribute("nlp.alert_count", len(alerts))

        return {
            "environmental_keywords": env_hits,
            "social_keywords": social_hits,
            "governance_keywords": gov_hits,
            "sentiment": simple_sentiment(text),
            "alerts": alerts,
        }