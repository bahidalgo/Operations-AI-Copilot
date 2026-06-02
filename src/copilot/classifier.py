"""
classifier.py

Descripcion:
    Clasificador de tickets usando embeddings semanticos (sentence-transformers)
    + nearest neighbor search por similitud coseno. Predice únicamente el TIPO de
    ticket. Prioridad y escalamiento se determinan en agent.py a partir del
    JSON de politicas generado por generate_policies.py.

    Modelo: all-MiniLM-L6-v2 (80MB, CPU, ~15ms por ticket).
    Se carga una sola vez al construir el clasificador y queda en memoria.

Uso:
    from copilot.classifier import build_classifier, classify_ticket, load_jsonl
"""

import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """
    Descripción: Carga un archivo JSONL línea por línea.
    Args:
        path: ruta al archivo .jsonl.
    Returns:
        Lista de dicts con los registros cargados.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Cargados {len(records)} registros desde {path}")
    return records


def build_classifier(train_data: list[dict]) -> dict:
    """
    Descripción: Construye el clasificador cargando el modelo de embeddings
                 y codificando todos los ejemplos de train. El modelo queda
                 en memoria para reutilizarse en cada clasificación.
    Args:
        train_data: lista de dicts con campos 'text' y 'label_type'.
    Returns:
        Dict con 'model', 'embeddings' y 'labels' listos para clasificar.
    """
    if not train_data:
        raise ValueError("train_data esta vacío. Se necesita al menos 1 ejemplo.")

    from sentence_transformers import SentenceTransformer

    log.info("Cargando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts  = [t["text"] for t in train_data]
    labels = [t["label_type"] for t in train_data]

    log.info(f"Codificando {len(texts)} ejemplos de train...")
    embeddings = model.encode(texts, show_progress_bar=False)

    log.info(f"Clasificador listo: {len(train_data)} ejemplos, "
             f"dimensiones de embedding: {embeddings.shape[1]}")

    return {"model": model, "embeddings": embeddings, "labels": labels}


def classify_ticket(text: str, classifier: dict, k: int = 3,
                    confidence_threshold: float = 0.50) -> dict:
    """
    Descripción: Predice el tipo de un ticket nuevo usando cosine similarity
                 contra los embeddings de train. No calcula prioridad ni
                 escalamiento porque eso lo resuelve agent.py consultando
                 el JSON de politicas generado por generate_policies.py.
    Args:
        text: texto del ticket a clasificar.
        classifier: dict retornado por build_classifier.
        k: numero de vecinos mas cercanos a retornar (pero como referencia),
           la decisión la toma siempre el más cercano top_k[0].
        confidence_threshold: umbral bajo el cual se marca low_confidence.
    Returns:
        Dict con:
        - predicted_type: tipo de ticket predicho.
        - confidence_score: similitud coseno con el vecino mas cercano (0-1).
        - low_confidence: True si el score esta bajo el umbral.
        - top_k_matches: lista de (label, score) de los k vecinos mas cercanos.
    """
    log.info(f"Clasificando: '{text[:60]}...'")

    embedding  = classifier["model"].encode([text], show_progress_bar=False)
    sims       = cosine_similarity(embedding, classifier["embeddings"])[0]
    ranked     = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    top_k      = ranked[:k]

    best_idx, best_score = top_k[0]
    predicted_type       = classifier["labels"][best_idx]
    low_confidence       = bool(best_score < confidence_threshold)

    if low_confidence:
        log.warning(f"Confianza baja ({best_score:.3f} < {confidence_threshold}). "
                    f"agent.py debe llamar a llm.py.")

    result = {
        "predicted_type":   predicted_type,
        "confidence_score": round(float(best_score), 3),
        "low_confidence":   low_confidence,
        "top_k_matches":    [(classifier["labels"][i], round(float(s), 3))
                             for i, s in top_k],
    }

    log.info(f"Resultado: type={predicted_type} | "
             f"confidence={result['confidence_score']} | "
             f"low_confidence={low_confidence}")

    return result
