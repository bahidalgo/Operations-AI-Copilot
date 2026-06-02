"""
run.py

Descripción:
    Punto de entrada del sistema. Carga el agente una sola vez, muestra
    predicciones sobre el test set, y entra en modo interactivo.

    PREREQUISITO: ejecutar generate_policies.py antes de correr este script.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src python run.py
"""

import logging
from copilot.agent import Agent
from copilot.llm import draft_response
from copilot.classifier import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEST_PATH = "02_dataset/tickets/tickets_test.jsonl"


def print_result(result: dict, ticket_text: str) -> None:
    """
    Descripción: Imprime el resultado del agente y las respuestas generadas.
    Args:
        result: dict retornado por agent.process().
        ticket_text: texto original del ticket.
    Returns:
        None.
    """
    print(f"       Tipo      : {result['predicted_type']}")
    print(f"       Fuente    : {result['source']}")
    print(f"       Escalar   : {result['escalate']}")
    print(f"       Razon     : {result['reason']}")
    if result["policy"] and not result["policy"].get("abstain"):
        print(f"       Prioridad : {result['policy'].get('priority')}")
        print(f"       Derivar a : {result['policy'].get('route_to')}")
    if result["policy"]:
        drafts = draft_response(ticket_text, result["policy"])
        print(f"       [Cliente]  : {drafts['client']}")
        print(f"       [Operador] : {drafts['operator']}")


def main() -> None:
    """
    Descripción: Inicializa el agente, procesa test set y entra en modo
                 interactivo. El agente falla explícitamente si
                 outputs/policies.json NO existe.
    Args:
        None.
    Returns:
        None.
    """
    try:
        agent = Agent()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    # Test set
    test = load_jsonl(TEST_PATH)
    print(f"\n{'='*60}\nPREDICCIONES SOBRE TEST SET ({len(test)} tickets)\n{'='*60}")
    for i, t in enumerate(test):
        result = agent.process(t["text"])
        print(f"\n  [{i+1}] {t['text']}")
        print_result(result, t["text"])

    # Modo interactivo
    print(f"\n{'='*60}\nMODO INTERACTIVO\n{'='*60}")
    print("Escribe un ticket o 'exit' para terminar.\n")

    while True:
        text = input("Ticket: ").strip()
        if not text:
            continue
        if text.lower() == "exit":
            break
        result = agent.process(text)
        print_result(result, text)
        print()


if __name__ == "__main__":
    main()
