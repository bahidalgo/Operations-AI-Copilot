"""
llm.py

Descripción:
    Clasificador de tickets usando LLM (Ollama) con few-shot prompting.
    También genera respuestas para cliente y operador via draft_response.

Uso:
    from copilot.llm import classify_with_llm, draft_response
"""

import json
import logging
import ollama

log       = logging.getLogger(__name__)
LLM_MODEL = "llama3.2:3b"


def classify_with_llm(text: str, ticket_types: list[str],
                      train_data: list[dict]) -> dict:
    """
    Descripción: Predice el tipo de un ticket usando LLM con few-shot
                 prompting. Los ejemplos del train se pasan como contexto
                 para clasificar por analogía.
    Args:
        text: texto del ticket a clasificar.
        ticket_types: lista de tipos válidos extraídos del train.
        train_data: ejemplos del train como contexto few-shot.
    Returns:
        Dict con:
        - predicted_type: tipo de ticket predicho.
        - failed: True si el LLM no retorno un tipo valido.
    """
    log.info(f"Clasificando con LLM: '{text[:60]}...'")

    types_str = ", ".join(ticket_types)
    examples  = "\n".join([
        f"- {t['label_type']}: \"{t['text']}\""
        for t in train_data
    ])

    prompt = f"""You are an operations classifier. Here are examples of each ticket type:

{examples}

Classify the following ticket into exactly one of these types: {types_str}.

Ticket: {text}

Respond ONLY with a JSON object with this exact format:
{{"predicted_type": "<type>"}}

Use only one of the provided types. No explanation, no markdown."""

    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={"temperature": 0}
    )
    raw = response["response"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed         = json.loads(raw)
        predicted_type = parsed.get("predicted_type", "").strip()
        failed         = predicted_type not in ticket_types

        if failed:
            log.warning(f"LLM retorno tipo desconocido: '{predicted_type}'.")
            predicted_type = None

    except json.JSONDecodeError as e:
        log.error(f"Error parseando respuesta del LLM: {e}\nRespuesta: {raw}")
        predicted_type = None
        failed         = True

    result = {
        "predicted_type": predicted_type,
        "failed":         failed,
    }

    log.info(f"Resultado LLM: type={predicted_type} | failed={failed}")
    return result


def draft_response(ticket_text: str, policy: dict) -> dict:
    """
    Descripción: Genera dos respuestas para un ticket procesado.

    Caso 1: política documentada (abstain=False):
        Cliente: informa que el equipo correspondiente se comunicará.
        Operador: instrucción directa con urgencia según prioridad y SLA.

    Caso 2 : sin política documentada (abstain=True):
        Cliente: mensaje genérico, un ejecutivo se comunicará.
        Operador: revisar manualmente, no hay política documentada.

    Args:
        ticket_text: texto original del ticket.
        policy: dict con flujo de acción retornado por el agente.
    Returns:
        Dict con:
        - client: mensaje breve para el cliente.
        - operator: instrucción directa para el operador interno.
    """
    log.info("Generando respuestas con LLM...")

    route_to  = policy.get("route_to") or "the appropriate team"
    priority  = policy.get("priority") or "medium"
    abstain   = policy.get("abstain", False)
    sla_times = {"high": "30 minutes", "medium": "4 hours", "low": "1 business day"}
    sla       = sla_times.get(priority, "as soon as possible")

    if abstain:
        client_prompt = f"""Write a brief, professional 2-sentence response to this customer ticket.
Inform them that an executive will contact them shortly. Be empathetic.

Ticket: {ticket_text}

Response to customer:"""

        operator_prompt = f"""Write a brief 1-sentence internal instruction.
Inform the team this case has no documented policy and requires manual review.

Ticket: {ticket_text}

Internal instruction:"""

    else:
        instructions = policy.get("instructions", "Follow standard procedure.")

        client_prompt = f"""Write a brief, professional 2-sentence response to this customer ticket.
Inform them that the {route_to} team will contact them. Be empathetic.

Ticket: {ticket_text}

Response to customer:"""

        operator_prompt = f"""Write a brief 1-sentence internal instruction with urgency.
Priority is {priority.upper()} — response required within {sla}.
Route to: {route_to}.
Policy: {instructions}

Ticket: {ticket_text}

Internal instruction:"""

    client_response = ollama.generate(
        model=LLM_MODEL,
        prompt=client_prompt,
        options={"temperature": 0, "num_predict": 80}
    )["response"].strip()

    operator_response = ollama.generate(
        model=LLM_MODEL,
        prompt=operator_prompt,
        options={"temperature": 0, "num_predict": 60}
    )["response"].strip()

    log.info("Respuestas generadas.")

    return {
        "client":   client_response,
        "operator": operator_response,
    }
