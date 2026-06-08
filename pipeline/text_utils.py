import re


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().strip())
