import re


def normalize_topic(topic: str) -> str:
    text = topic.casefold().strip()

    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    aliases = {
        "eigen value": "eigenvalues",
        "eigen values": "eigenvalues",
        "eigenvalue": "eigenvalues",
        "eigenvalues": "eigenvalues",
    }

    return aliases.get(text, text)
