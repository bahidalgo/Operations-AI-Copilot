"""
policy.py

Descripción:
    Lee los documentos internos, filtra los vigentes por Status: Active,
    y usa un LLM para generar un diccionario de flujos por tipo de ticket.
    El resultado se guarda en outputs/policies.json para ser cargado
    por el agente en runtime.

    Documentos con Status Deprecated, Outdated o Mixed reliability
    son ignorados automaticamente.

    Nota: la deteccion de conflictos entre documentos Active no esta
    implementada. Si dos documentos Active contienen instrucciones
    contradictorias, el LLM recibira contexto inconsistente y el humano
    que revise el json DEBE resolver el conflicto manualmente.

Uso:
    from copilot.policy import load_policies
    policies = load_policies("02_dataset/docs", ticket_types)
"""

import os
import json
import logging
import ollama

log = logging.getLogger(__name__)

ACTIVE_STATUSES = {"active"}
OLLAMA_MODEL    = "llama3.2:3b"


def _read_docs(docs_path: str) -> list[dict]:
    """
    Descripci+ón: Lee todos los archivos .md del directorio y extrae
                 nombre, status y contenido de cada uno.
    Args:
        docs_path: ruta al directorio con los documentos.
    Returns:
        Lista de dicts con 'filename', 'status' y 'content'.
    """
    docs = []
    for filename in sorted(os.listdir(docs_path)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(docs_path, filename)
        with open(path) as f:
            content = f.read()
        status = "unknown"
        for line in content.splitlines():
            if line.lower().startswith("status:"):
                status = line.split(":", 1)[1].strip().lower()
                break
        docs.append({"filename": filename, "status": status, "content": content})
        log.info(f"Documento leído: {filename} | status: {status}")
    return docs


def _filter_active(docs: list[dict]) -> list[dict]:
    """
    Descripcion: Retorna solo los documentos con status Active.
                 Documentos Deprecated, Outdated o de confiabilidad mixta
                 son ignorados.
    Args:
        docs: lista de dicts con 'filename', 'status' y 'content'.
    Returns:
        Lista de documentos vigentes.
    """
    active  = [d for d in docs if d["status"] in ACTIVE_STATUSES]
    ignored = [d["filename"] for d in docs if d["status"] not in ACTIVE_STATUSES]
    if ignored:
        log.info(f"Documentos ignorados: {ignored}")
    log.info(f"Documentos vigentes: {[d['filename'] for d in active]}")
    return active


def _build_context(active_docs: list[dict]) -> str:
    """
    Descripción: Concatena el contenido de los documentos vigentes
                 en un solo string para pasarlo al LLM.
    Args:
        active_docs: lista de documentos vigentes.
    Returns:
        String con todo el contenido concatenado y separado por documento.
    """
    parts = []
    for d in active_docs:
        parts.append(f"--- {d['filename']} ---\n{d['content']}")
    return "\n\n".join(parts)


def _generate_policies(context: str, ticket_types: list[str]) -> dict:
    """
    Descripción: Llama al LLM con el contexto de documentos vigentes y
                 los tipos de ticket. El LLM genera un flujo de acción
                 para cada tipo en formato JSON.
    Args:
        context: string con el contenido de todos los documentos vigentes.
        ticket_types: lista de tipos de ticket conocidos, extraídos del train.
    Returns:
        Dict con un flujo por tipo de ticket.
    """
    import re
    types_str = ", ".join(ticket_types)

    prompt = f"""You are an operations assistant. Read the following internal documents and generate a SINGLE JSON object with an action flow for ALL ticket types combined.

DOCUMENTS:
{context}

TICKET TYPES: {types_str}

Instructions:
- Read all documents carefully before deciding.
- Set abstain to false if ANY document covers the ticket type, even indirectly.
- Determine escalate based on what the documents explicitly require.
- Use the SLA matrix for priority when available.
- Set route_to based on escalation targets mentioned in the documents.

Return ONE single JSON object containing ALL ticket types. Each ticket type must have:
- priority: "high", "medium", or "low"
- escalate: true or false
- route_to: responsible team (string) or null
- instructions: one sentence from the documents
- abstain: true or false

Respond ONLY with ONE valid JSON object containing all types. No markdown, no explanation, no separate objects.

Example format:
{{
  "type_a": {{
    "priority": "medium",
    "escalate": true,
    "route_to": "team name",
    "instructions": "What to do.",
    "abstain": false
  }},
  "type_b": {{
    "priority": "low",
    "escalate": false,
    "route_to": null,
    "instructions": "What to do.",
    "abstain": true
  }}
}}"""

    log.info("Llamando al LLM para generar políticas...")
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0})
    raw      = response["response"].strip()
    log.info("Respuesta recibida del LLM.")

    # Extraer bloque JSON aunque el LLM agregue texto adicional
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group()
    else:
        log.error(f"No se encontro JSON en la respuesta del LLM:\n{raw}")
        raise ValueError("El LLM no retorno un JSON válido.")

    try:
        policies = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"Error parseando respuesta del LLM: {e}\nRespuesta: {raw}")
        raise

    return policies


def load_policies(docs_path: str, ticket_types: list[str]) -> dict:
    """
    Descripción: Función principal. Lee los documentos, filtra los vigentes,
                 genera flujos con el LLM y retorna el diccionario de políticas.
                 Debe llamarse una sola vez al arrancar el sistema.
    Args:
        docs_path: ruta al directorio con los documentos internos.
        ticket_types: lista de tipos conocidos, extraídos del dataset de train.
                      No se asume ningún tipo fijo.
    Returns:
        Dict con un flujo de acción por tipo de ticket. Ejemplo:
        {
            "pricing_dispute": {
                "priority": "medium",
                "escalate": true,
                "route_to": "pricing operations",
                "instructions": "Route to pricing operations for review.",
                "abstain": false
            }
        }
    """
    if not ticket_types:
        raise ValueError("ticket_types esta vacio. Se necesita al menos un tipo.")

    docs        = _read_docs(docs_path)
    active_docs = _filter_active(docs)

    if not active_docs:
        raise ValueError(f"No se encontraron documentos vigentes en {docs_path}")

    context  = _build_context(active_docs)
    policies = _generate_policies(context, ticket_types)

    log.info(f"Políticas generadas para tipos: {list(policies.keys())}")
    return policies
