"""
test_policy.py

Descripción:
    Tests de policy.py. Verifica disponibilidad del LLM, que el filtro
    de documentos funciona correctamente, y que load_policies retorna
    la estructura esperada para todos los tipos del train.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src pytest tests/test_policy.py -v
"""

import pytest
import ollama
from copilot.policy import _filter_active, load_policies
from copilot.classifier import load_jsonl

DOCS_PATH  = "02_dataset/docs"
TRAIN_PATH = "02_dataset/tickets/tickets_train.jsonl"


def test_llm_disponible():
    """
    Descripción: verifica que Ollama esta corriendo y el modelo responde.
                 Si este test falla, los demás tests que usan LLM fallarán también.
    """
    try:
        response = ollama.generate(model="llama3.2:3b", prompt="say ok")
        assert response["response"] is not None
    except Exception as e:
        pytest.fail(f"Ollama no esta disponible: {e}")


def test_filter_active_ignora_no_vigentes():
    """
    Descripción: documentos con status Deprecated, Outdated o desconocido
                 deben ser ignorados. Solo los Active deben pasar.
    """
    docs = [
        {"filename": "policy_v1.md",   "status": "deprecated", "content": "old"},
        {"filename": "policy_v2.md",   "status": "active",     "content": "current"},
        {"filename": "pricing_old.md", "status": "outdated",   "content": "old"},
        {"filename": "faq.md",         "status": "mixed reliability", "content": "faq"},
    ]
    result = _filter_active(docs)
    assert len(result) == 1
    assert result[0]["filename"] == "policy_v2.md"


def test_load_policies_retorna_flujo_para_todos_los_tipos():
    """
    Descripción: load_policies debe retornar un flujo con los campos
                 requeridos para cada tipo extraido del dataset de train.
    """
    train        = load_jsonl(TRAIN_PATH)
    ticket_types = list({t["label_type"] for t in train})
    policies     = load_policies(DOCS_PATH, ticket_types)

    for ticket_type in ticket_types:
        assert ticket_type in policies, f"Falta flujo para tipo: {ticket_type}"
        assert "priority"     in policies[ticket_type]
        assert "escalate"     in policies[ticket_type]
        assert "instructions" in policies[ticket_type]
        assert "abstain"      in policies[ticket_type]
