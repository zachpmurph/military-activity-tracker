from core.config import classify_aircraft


def compute_score(a, behavior):
    source_count = len(a.get("sources", []))
    callsign = a["callsign"]
    altitude = a["altitude"] or 0
    speed = a["velocity"] or 0

    score = 0
    score += source_count
    classification = classify_aircraft(a)

    if classification == "CIVILIAN":
        score -= 2
    elif classification == "UNKNOWN":
        score += 0
    if classification in ("US_CARGO", "TANKER", "UK_CARGO"):
        score += 2
    if not callsign:
        score -= 2

    if callsign:
        score += 1

    if altitude > 25000:
        score += 1

    if altitude > 30000 and speed < 250:
        score += 2

    if behavior == "LOITERING":
        score += 2

    if behavior == "INSUFFICIENT_DATA":
        score -= 1

    return score
