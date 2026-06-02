"""
generate_policies.py

Descripción:
    Lee los documentos internos, filtra los vigentes, y usa el LLM para
    generar un diccionario de flujos por tipo de ticket. Guarda el resultado
    en outputs/policies.json.

    Ejecutar manualmente cada vez que se agreguen, modifiquen o eliminen
    documentos en 02_dataset/docs/.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src python generate_policies.py
"""

import json
import logging
import os
from copilot.policy import _read_docs, _filter_active, load_policies
from copilot.classifier import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DOCS_PATH   = "02_dataset/docs"
TRAIN_PATH  = "02_dataset/tickets/tickets_train.jsonl"
OUTPUT_PATH = "outputs/policies.json"


def main() -> None:
    """
    Descripcion: Muestra documentos encontrados, filtro de vigencia,
                 políticas generadas, y guarda el resultado en JSON.
                 Si abstain es True para un tipo, limpia los campos
                 para evitar instrucciones inventadas por el LLM.
    Args:
        None.
    Returns:
        None.
    """
    # 1. Documentos encontrados
    print(f"\n{'='*60}\n1. DOCUMENTOS ENCONTRADOS\n{'='*60}")
    docs = _read_docs(DOCS_PATH)
    for d in docs:
        print(f"  {d['filename']:<35} status: {d['status']}")

    # 2. Filtro de vigencia
    print(f"\n{'='*60}\n2. FILTRO DE VIGENCIA\n{'='*60}")
    active  = _filter_active(docs)
    ignored = [d for d in docs if d not in active]

    print(f"\n  Vigentes ({len(active)}):")
    for d in active:
        print(f"    - {d['filename']}")

    print(f"\n  Descartados ({len(ignored)}):")
    for d in ignored:
        print(f"    - {d['filename']} (status: {d['status']})")

    unknown = [d for d in docs if d["status"] not in
               {"active", "deprecated", "outdated", "mixed reliability"}]
    if unknown:
        print(f"\n  [ATENCIÓN] Documentos con status no reconocido -> requieren revisión humana:")
        for d in unknown:
            print(f"    - {d['filename']} (status: {d['status']})")

    # 3. Generar políticas
    print(f"\n{'='*60}\n3. POLÍTICAS GENERADAS POR LLM\n{'='*60}")
    train        = load_jsonl(TRAIN_PATH)
    ticket_types = list({t["label_type"] for t in train})
    print(f"\n  Tipos extraidos del train: {ticket_types}\n")

    policies = load_policies(DOCS_PATH, ticket_types)

    # Si abstain es True, limpiar campos para evitar instrucciones inventadas
    for ticket_type, flow in policies.items():
        if flow.get("abstain") is True:
            flow["priority"]     = None
            flow["escalate"]     = True
            flow["route_to"]     = None
            flow["instructions"] = None

    for ticket_type, flow in policies.items():
        print(f"  [{ticket_type}]")
        print(f"    priority    : {flow.get('priority')}")
        print(f"    escalate    : {flow.get('escalate')}")
        print(f"    route_to    : {flow.get('route_to')}")
        print(f"    instructions: {flow.get('instructions')}")
        print(f"    abstain     : {flow.get('abstain')}")
        print()

    #4. Guardar JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(policies, f, indent=2)
    print(f"Politicas guardadas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
