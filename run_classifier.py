"""
run_classifier.py

Descripción:
    Punto de entrada del clasificador de tickets. Carga el modelo una sola
    vez, muestra predicciones sobre el test set, y luego entra en modo
    interactivo donde podemos escribir nuestros propios tickets.

Uso (desde la raiz del proyecto):
    PYTHONPATH=src python run.py

Salida:
    Predicciones del test set y luego prompt interactivo.
"""

import logging
from copilot.classifier import load_jsonl, build_classifier, classify_ticket

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = "02_dataset/tickets/tickets_train.jsonl"
TEST_PATH  = "02_dataset/tickets/tickets_test.jsonl"


def print_result(result: dict) -> None:
    """
    Descripción: Imprime el resultado de clasificación de un ticket.
    Args:
        result: dict retornado por classify_ticket.
    Returns:
        None.
    """
    print(f"       Tipo      : {result['predicted_type']}")
    print(f"       Confianza : {result['confidence_score']}")
    print(f"       Top-3     : {result['top_k_matches']}")
    if result["low_confidence"]:
        print(f"       [AVISO] Confianza baja, requiere revision humana")


def main() -> None:
    """
    Descripción: Carga el clasificador, procesa el test set y entra en modo
                 interactivo. El modelo se carga una sola vez al inicio.
    Args:
        None.
    Returns:
        None.
    """
    train      = load_jsonl(TRAIN_PATH)
    classifier = build_classifier(train)

    # Primero lo haremos funcionar con el Test set
    test = load_jsonl(TEST_PATH)
    print(f"\n{'='*60}\nPREDICCIONES SOBRE TEST SET ({len(test)} tickets)\n{'='*60}")
    for i, t in enumerate(test):
        result = classify_ticket(t["text"], classifier)
        print(f"\n  [{i+1}] {t['text']}")
        print_result(result)

    # Luego, podemos hacer pruebas con el modo interactivo
    print(f"\n{'='*60}\nMODO INTERACTIVO\n{'='*60}")
    print("Escribe un ticket o 'exit' para terminar.\n")

    while True:
        text = input("Ticket: ").strip()
        if not text:
            continue
        if text.lower() == "exit":
            break
        result = classify_ticket(text, classifier)
        print_result(result)
        print()


if __name__ == "__main__":
    main()
