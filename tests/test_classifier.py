"""
test_classifier.py

Descripción:
    Tests del clasificador con embeddings semánticos.
    Cubre: existencia y formato de archivos, contrato de la función,
    y guardrail de confianza baja.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src pytest tests/test_classifier.py -v
"""

import os
import pytest
from copilot.classifier import build_classifier, classify_ticket, load_jsonl


@pytest.fixture
def minimal_train():
    """
    Dataset mínimo hardcodeado para tests unitarios que no dependen
    de archivos externos. Permite correr los tests sin el dataset real.
    """
    return [
        {"text": "Customer reports reefer container temperature deviation.",
         "label_type": "temperature_exception"},
        {"text": "Customer asks where the shipment is and requests ETA update.",
         "label_type": "tracking_request"},
        {"text": "Customer says invoice amount differs from quoted price.",
         "label_type": "pricing_dispute"},
        {"text": "Customer wants clarification on contract liability.",
         "label_type": "legal_sensitive"},
        {"text": "Customer asks for copy of old invoice.",
         "label_type": "document_request"},
        {"text": "Customer reports customs retention.",
         "label_type": "shipment_exception"},
    ]


@pytest.fixture
def classifier(minimal_train):
    return build_classifier(minimal_train)


# Tests de dataset

def test_archivos_dataset_existen():
    """Descripción: verifica que los archivos de train y test existen en la ruta esperada."""
    assert os.path.exists("02_dataset/tickets/tickets_train.jsonl"), \
        "No se encontro tickets_train.jsonl en 02_dataset/tickets/"
    assert os.path.exists("02_dataset/tickets/tickets_test.jsonl"), \
        "No se encontro tickets_test.jsonl en 02_dataset/tickets/"


def test_train_tiene_campos_requeridos():
    """Descripción: verifica que cada registro de train tenga text y label_type."""
    train = load_jsonl("02_dataset/tickets/tickets_train.jsonl")
    assert len(train) > 0, "El archivo de train esta vacío"
    for i, record in enumerate(train):
        assert "text" in record, f"Registro {i+1} no tiene campo 'text'"
        assert "label_type" in record, f"Registro {i+1} no tiene campo 'label_type'"


# Tests del clasificador

def test_vacio_lanza_error():
    """Descripción: train vacío debe lanzar error explícito."""
    with pytest.raises(ValueError):
        build_classifier([])


def test_retorna_campos_requeridos(classifier):
    """Descripción: el resultado debe tener los campos que el resto del sistema consume."""
    result = classify_ticket("temperature alert in reefer cargo", classifier)
    for field in ["predicted_type", "confidence_score", "low_confidence", "top_k_matches"]:
        assert field in result


def test_tipo_predicho_es_label_conocido(minimal_train, classifier):
    """Descripción: el tipo predicho debe pertenecer al conjunto de labels de train."""
    known  = {t["label_type"] for t in minimal_train}
    result = classify_ticket("customer asks for ETA update", classifier)
    assert result["predicted_type"] in known


def test_low_confidence_ticket_vago(classifier):
    """Descripción: un ticket vago y mal escrito debe activar el guardrail con el umbral real."""
    result = classify_ticket("I have a problem. I need that return mi package",
                             classifier, confidence_threshold=0.50)
    assert result["low_confidence"] is True
