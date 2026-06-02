"""
agent.py

Descripción:
    Orquestador principal del sistema. Recibe un ticket y ejecuta el flujo
    completo, clasificación con embeddings, respaldo con LLM si hay baja
    confianza, y lookup de política vigente en el JSON pregenerado.

    El agente NO genera políticas  solo las carga. Si outputs/policies.json
    no existe, falla con instrucciones claras para el operador.

Uso:
    from copilot.agent import Agent
    agent = Agent()
    result = agent.process("Where is my package?")
"""

import json
import logging
import os

log = logging.getLogger(__name__)

POLICIES_PATH      = "outputs/policies.json"
TRAIN_PATH         = "02_dataset/tickets/tickets_train.jsonl"
CONFIDENCE_THRESHOLD = 0.60


class Agent:

    def __init__(self):
        """
        Descripción: Carga el clasificador de embeddings, los tipos de ticket
                     del train, y las políticas del JSON pregenerado.
                     Falla explícitamente si policies.json no existe.
        """
        from copilot.classifier import load_jsonl, build_classifier
        from copilot.llm import classify_with_llm

        # Verificar que policies.json existe
        if not os.path.exists(POLICIES_PATH):
            raise FileNotFoundError(
                f"No se encontró {POLICIES_PATH}.\n"
                f"Ejecuta primero: python generate_policies.py\n"
                f"Revisa el archivo generado antes de correr el sistema."
            )

        # Cargar train y extraer tipos
        train             = load_jsonl(TRAIN_PATH)
        self.train_data   = train
        self.ticket_types = list({t["label_type"] for t in train})
        log.info(f"Tipos de ticket conocidos: {self.ticket_types}")

        # Construir clasificador de embeddings
        self.classifier = build_classifier(train)

        # Cargar politicas
        with open(POLICIES_PATH) as f:
            self.policies = json.load(f)
        log.info(f"Políticas cargadas desde {POLICIES_PATH}")

        self._classify_with_llm = classify_with_llm

    def process(self, text: str) -> dict:
        """
        Descripción: Procesa un ticket ejecutando el flujo completo.
             1. Clasificar con embeddings
             2. Si confianza < umbral, clasificar con LLM 
             3. Buscar política en JSON según tipo predicho
             4. Generar respuestas para cliente y operador
             5. Retornar resultado con flujo a seguir

             NOTA: el LLM raramente falla (solo si el JSON es inválido).
             El guardrail de escalamiento por falla del LLM existe como
             red de seguridad pero no es el caso habitual, porque se espera
             que el json sea revisado por un humano.

        Args:
            text: texto del ticket ingresado por el operador.
        Returns:
            Dict con:
            - predicted_type: tipo predicho.
            - source: 'embeddings' o 'llm' o 'human_escalation'.
            - policy: dict con flujo de acción, o None si escala.
            - escalate: True si debe escalar a humano.
            - reason: motivo de la decisión.
        """
        from copilot.classifier import classify_ticket

        log.info(f"Procesando ticket: '{text[:60]}...'")

        # Paso 1: clasificar con embeddings
        emb_result = classify_ticket(text, self.classifier,
                                     confidence_threshold=CONFIDENCE_THRESHOLD)

        if not emb_result["low_confidence"]:
            predicted_type = emb_result["predicted_type"]
            source         = "embeddings"
            log.info(f"Clasificado con embeddings: {predicted_type} "
                     f"(confidence={emb_result['confidence_score']})")

        else:
            # Paso 2: respaldo con LLM
            log.info(f"Confianza baja ({emb_result['confidence_score']}). "
                     f"Intentando con LLM...")
            llm_result = self._classify_with_llm(text, self.ticket_types, self.train_data)

            if not llm_result["failed"]:
                predicted_type = llm_result["predicted_type"]
                source         = "llm"
                log.info(f"Clasificado con LLM: {predicted_type}")
            else:
                # Paso 3: escalar a humano
                log.warning("LLM no pudo clasificar. Escalando a humano.")
                return {
                    "predicted_type": None,
                    "source":         "human_escalation",
                    "policy":         None,
                    "escalate":       True,
                    "reason":         "Clasificador y LLM no lograron determinar el tipo.",
                }

        # Paso 4: buscar política
        policy = self.policies.get(predicted_type)

        if not policy:
            log.warning(f"Tipo '{predicted_type}' no encontrado en politicas.")
            return {
                "predicted_type": predicted_type,
                "source":         source,
                "policy":         None,
                "escalate":       True,
                "reason":         f"Tipo '{predicted_type}' no tiene politica definida.",
            }

        # Paso 5: evaluar política
        if policy.get("abstain"):
            reason   = f"Política indica abstenerse para tipo '{predicted_type}'."
            escalate = True
        elif policy.get("escalate"):
            reason   = policy.get("instructions", "Escalar segun política.")
            escalate = True
        else:
            reason   = policy.get("instructions", "Responder segun política.")
            escalate = False

        result = {
            "predicted_type": predicted_type,
            "source":         source,
            "policy":         policy,
            "escalate":       escalate,
            "reason":         reason,
        }

        log.info(f"Resultado: type={predicted_type} | source={source} | "
                 f"escalate={escalate}")
        return result
