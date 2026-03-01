import re
from typing import List


def clean_text(s: str) -> str:
    """Limpieza ligera (no agresiva): trim + colapsar espacios."""
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_for_model(texts: List[str], mode: str, model_name: str) -> List[str]:
    """
    Para modelos tipo E5/BGE es importante usar prefijos:
      - query: ...   (cuando el texto actúa como consulta)
      - passage: ... (cuando el texto actúa como documento)

    mode debe ser: "query" o "passage"
    """
    model_lower = (model_name or "").lower()
    if ("e5" in model_lower) or ("bge" in model_lower):
        prefix = "query: " if mode == "query" else "passage: "
        return [prefix + t for t in texts]
    return texts
