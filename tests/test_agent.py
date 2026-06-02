"""
test_agent.py

Descripción:
    Tests del agente. Verifica que falla correctamente si no existe el JSON
    de políticas, y que el flujo completo retorna la estructura esperada.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src pytest tests/test_agent.py -v
"""

import pytest
import os
from copilot.agent import Agent


def test_agent_falla_sin_policies_json(tmp_path, monkeypatch):
    """
    Descripción: el agente debe fallar si outputs/policies.json no existe.
                 Ningún ticket debe procesarse sin políticas validadas.
    """
    monkeypatch.setattr("copilot.agent.POLICIES_PATH", str(tmp_path / "policies.json"))
    with pytest.raises(FileNotFoundError):
        Agent()


def test_agent_retorna_campos_requeridos():
    """
    Descripción: el flujo completo debe retornar siempre los campos que
                 run.py y el resto del sistema consumen.
    """
    agent  = Agent()
    result = agent.process("Customer reports temperature alert in reefer cargo.")
    for field in ["predicted_type", "source", "policy", "escalate", "reason"]:
        assert field in result, f"Campo faltante: {field}"
