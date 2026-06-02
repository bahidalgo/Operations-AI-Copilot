"""
run_llm.py

Descripción:
    Muestra predicciones del clasificador LLM sobre el test set y luego
    entra en modo interactivo. Análogo a run_classifier.py pero usando
    el LLM con few-shot prompting en lugar de embeddings.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src python -m copilot.run_llm
"""

import logging
from copilot.llm import classify_with_llm
from copilot.classifier import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TRAIN_PATH = "02_dataset/tickets/tickets_train.jsonl"
TEST_PATH  = "02_dataset/tickets/tickets_test.jsonl"


def print_result(result: dict) -> None:
    """
    Descripción: Imprime el resultado de clasificación de un ticket.
    Args:
        result: dict retornado por classify_with_llm.
    Returns:
        None.
    """
    print(f"       Tipo      : {result['predicted_type']}")
    if result["failed"]:
        print(f"       [AVISO] LLM no pudo clasificar -> requiere revisión humana")


def main() -> None:
    """
    Descripcion: Carga los tipos del train, procesa el test set con el LLM
                 y entra en modo interactivo.
    Args:
        None.
    Returns:
        None.
    """
    train        = load_jsonl(TRAIN_PATH)
    ticket_types = list({t["label_type"] for t in train})

    # ── Test set ──────────────────────────────────────────────────────────────
    test = load_jsonl(TEST_PATH)
    print(f"\n{'='*60}\nPREDICCIONES LLM SOBRE TEST SET ({len(test)} tickets)\n{'='*60}")
    for i, t in enumerate(test):
        result = classify_with_llm(t["text"], ticket_types, train)
        print(f"\n  [{i+1}] {t['text']}")
        print_result(result)

    # ── Modo interactivo ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nMODO INTERACTIVO\n{'='*60}")
    print("Escribe un ticket o 'salir' para terminar.\n")

    while True:
        text = input("Ticket: ").strip()
        if not text:
            continue
        if text.lower() == "salir":
            break
        result = classify_with_llm(text, ticket_types, train)
        print_result(result)
        print()


if __name__ == "__main__":
    main()
