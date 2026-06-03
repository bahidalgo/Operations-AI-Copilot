# Operations AI Copilot

**Autor:** Byron Hidalgo

Sistema de clasificación y enrutamiento de tickets operativos para equipos de logística. Combina un clasificador de embeddings semánticos, un LLM local de respaldo, y un sistema de políticas generadas a partir de documentación interna vigente.

---

## Resumen de la solución

El sistema recibe un ticket de cliente en lenguaje natural y retorna:
- el tipo de ticket clasificado
- la fuente de la clasificacion (embeddings o LLM)
- si debe escalarse a un humano
- a quién derivarlo y por qué
- una respuesta borrador para el cliente y una instrucción para el operador interno

El flujo es híbrido. Un clasificador rápido y liviano actúa primero y el LLM interviene solo cuando la confianza es insuficiente. Las políticas de negocio se generan a partir de los documentos internos vigentes mediante un módulo independiente que genera un json, el cual debe ser revisado por un humano antes de usar en producción.

---

## Arquitectura

```
generate_policies.py     (se ejecuta una vez, revisar manualmente)
        |
        v
outputs/policies.json    (políticas que deben ser validadas por humano)
        |
        v
run.py
  |
  |-- Agent.__init__()
  |     |-- load_jsonl()          carga train, extrae ticket_types
  |     |-- build_classifier()    carga modelo embeddings (all-MiniLM-L6-v2)
  |     |-- load policies.json    carga políticas pregeneradas
  |
  |-- Agent.process(ticket)
        |-- classify_ticket()     embeddings + cosine similarity
        |     confianza >= 0.60 ---> predicted_type
        |     confianza <  0.60 ---> classify_with_llm()
        |                               predicted_type
        |                               (guardrail: si JSON invalido -> escalar a humano)
        |
        |-- lookup policies[predicted_type]
        |     abstain=True  --> no hay política documentada, escalar a humano
        |     escalate=True --> derivar segun route_to
        |     escalate=False --> responder automáticamente (para una versión futura)
        |
        |-- draft_response()      genera respuesta para cliente y operador
              abstain=True  --> cliente: un ejecutivo se comunicará
                               operador: revisar manualmente, sin política documentada
              abstain=False --> cliente: equipo correspondiente se comunicará
                               operador: instrucción con urgencia según prioridad y SLA
```

---

## Estructura del proyecto

```
AI-Engineer-Testing/
├── generate_policies.py     script de mantenimiento, genera outputs/policies.json
├── run.py                   punto de entrada principal
├── requirements.txt
├── .gitignore
├── src/copilot/
│   ├── classifier.py        clasificador con embeddings semánticos
│   ├── llm.py               clasificador LLM de respaldo y draft_response (Ollama)
│   ├── policy.py            lectura y filtrado de documentos internos
│   ├── agent.py             orquestador del flujo completo
│   ├── run_classifier.py    demo del clasificador de embeddings
│   └── run_llm.py           demo del clasificador LLM
├── tests/
│   ├── test_classifier.py
│   ├── test_policy.py
│   └── test_agent.py
└── 02_dataset/
    ├── docs/                documentos internos de políticas
    └── tickets/             train y test sets
```

---

## Instalación y uso

### Prerequisitos

- Python 3.12
- Ollama instalado y corriendo con el modelo `llama3.2:3b`

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo
ollama pull llama3.2:3b
```

### Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Uso

```bash
# 1. Generar politicas (ejecutar cada vez que cambien los documentos)
#    Revisar outputs/policies.json antes de continuar
PYTHONPATH=src python generate_policies.py

# 2. Correr el sistema
PYTHONPATH=src python run.py

# 3. Correr tests
PYTHONPATH=src pytest tests/ -v
```

---

## Decisiones técnicas

**Por qué embeddings y no TF-IDF:** TF-IDF no captura semántica, por ejemplo "refrigerated cargo" y "reefer deviation" son distintos para TF-IDF pero similares en significado. Los embeddings resuelven esto sin necesidad de más datos.

**Por qué Ollama local y no API externa:** presupuesto limitado y sin dependencia de conectividad. El costo por llamada es cero y el modelo queda en la infraestructura propia.

**Por qué el JSON de políticas se genera offline:** las políticas son decisiones de negocio que un humano debe validar. Generarlas en runtime cada vez sería costoso, lento, y quitaría control al operador.

**Por qué umbral de confianza 0.60:** ajustado empíricamente, pues con 0.50 algunos casos ambiguos pasaban con clasificación incorrecta. Un análisis de distribución con más datos permitiría ajustarlo con mayor rigor.

---

## Enfoque de ML

### Clasificador principal | embeddings semanticos

Modelo: `all-MiniLM-L6-v2` (sentence-transformers, 80MB, CPU/GPU).
Approach: cosine similarity sobre los embeddings del train set.

El modelo se carga una sola vez al arrancar y queda en memoria. Cada ticket nuevo se codifica y se compara contra todos los ejemplos de train. El tipo del vecino más cercano es la predicción.

Se eligió este modelo porque:
- Funciona bien con pocos ejemplos de train (no requiere cientos de datos por clase)
- Captura semántica -> "refrigerated cargo" y "reefer deviation" son similares aunque no compartan palabras
- Es rápido en inferencia (~15ms por ticket en CPU, menos en GPU)

**Umbral de confianza:** 0.60. Si la similitud coseno con el vecino mas cercano es menor a este valor, la prediccion se considera insuficiente y se activa fallback con LLM. Este umbral fue ajustado empíricamente, con 0.50 algunos casos ambiguos pasaban con clasificacion incorrecta, pero para una versión más estable, se recomienda hacer un análisis de distribución.

### Baseline | TF-IDF

Se implementó un baseline con TF-IDF + KNN como punto de comparación, pero los resultados mostraron la limitación de los métodos estadísticos con pocos datos.

### Clasificador de respaldo | LLM

Modelo: `llama3.2:3b` vía Ollama (local, sin API key, sin costo).
Se usa únicamente cuando el clasificador de embeddings tiene baja confianza.
Temperatura: 0 para minimizar variabilidad entre ejecuciones, aunque con modelos pequeños como llama3.2:3b, temperatura 0 no garantiza reproducibilidad en casos ambiguos.

---

## Métricas de evaluación

| Componente | Métrica | Resultado |
|---|---|---|
| Embeddings (sobre test) | Type accuracy | 5/6 (~83%) |
| LLM clasificador (sobre test) | Type accuracy | 100% (6/6) |
| Sistema completo con agente (sobre test) | Casos correctamente enrutados | 5/6 (~83%) |

---

## Enfoque agentic

### Por qué es agentic

El agente toma decisiones secuenciales donde el resultado de cada paso determina el siguiente:
1. Si confianza alta, usar embeddings directamente
2. Si confianza baja, invocar LLM
3. Si LLM falla, escalar a humano sin intentar más
4. Si politica dice abstain, escalar sin inventar respuesta

Esto es un flujo condicional donde el agente decide qué herramienta usar y cuándo parar.

### Que NO es agentic

La generacion de politicas (`generate_policies.py`) es un proceso batch que corre offline y requiere validación humana. No es parte del flujo agentic, es un prerequisito que un humano debe aprobar.

### Guardrails implementados

- **Umbral de confianza:** el agente no actua con confianza insuficiente
- **Abstain:** si los documentos no cubren un tipo, el sistema se abstiene en lugar de inventar
- **Validacion humana de políticas:** el JSON de politicas debe ser revisado antes de usarse en producción
- **Fallo explícito sin policies.json:** el sistema no debería arrancar sin políticas validadas. Por ahora se considera solo la generación, pero a futuro se debería añadir metadata asociado a revisado y/o validado.
- **LLM solo como respaldo:** el LLM no toma decisiones de negocio, solo clasifica cuando los embeddings fallan y decide a quién delegar.

---

## Manejo de documentos contradictorios y vigencia

Cada documento tiene un campo `Status` en su encabezado. El sistema solo procesa documentos con `Status: Active`. Documentos con `Status: Deprecated`, `Outdated` o `Mixed reliability` son ignorados automaticamente. En producción, la categoría `Mixed reliability` debería ser revisada y actualizada a Active o eliminada, es decir, debería considerarse revisión humana.

Los conflictos en este dataset son entre documentos vigentes y obsoletos, resueltos automáticamente por el filtro de vigencia. Si en producción aparecieran dos documentos `Active` con instrucciones contradictorias sobre el mismo tipo, el LLM recibiría contexto contradictorio y podría generar políticas inconsistentes. Por eso la revisión humana del JSON generado es obligatoria antes de usar en producción.

---

## Restricciones del desafío

**Presupuesto limitado:** todos los modelos corren localmente sin costo por llamada. Sin API key, sin dependencia de servicios externos.

**Latencia:** el clasificador de embeddings responde en ~15ms. El LLM solo se invoca cuando la confianza es baja, minimizando las llamadas lentas (~0.3s con GPU).

**Documentos contradictorios y desactualizados:** el filtro de vigencia descarta automáticamente documentos no Active. Los conflictos entre documentos Active requieren revisión humana del JSON generado.

**Usuarios que preguntan mal:** el LLM de respaldo maneja lenguaje informal, mezclado o con errores gramaticales mejor que el clasificador de embeddings, que depende de similitud léxica.

**Riesgo de autonomía:** el sistema no toma decisiones irreversibles de forma autónoma. Toda acción con escalamiento requiere un humano. Las políticas que determinan el flujo son validadas manualmente antes de entrar en producción.

---

## Observabilidad

Todos los módulos usan `logging` con timestamps y niveles `INFO` / `WARNING` / `ERROR`. Los logs muestran:
- Documentos cargados y su status
- Confianza de cada clasificacion
- Fuente de la prediccion (embeddings o LLM)
- Warnings cuando la confianza es baja
- Errors cuando el LLM no retorna JSON valido

---

## Costos estimados

| Componente | Costo por llamada | Costo fijo |
|---|---|---|
| Clasificador embeddings (all-MiniLM-L6-v2) | $0 | 100MB en disco |
| LLM clasificador (llama3.2:3b, Ollama) | $0 | 3GB en disco |
| Generacion de politicas | $0 | 4s de compute, una vez por cambio de docs con GPU|

El sistema no tiene costo variable por llamada, todos los modelos corren localmente.
El costo real es de infraestructura, el sistema requiere mínimo 8GB RAM para cargar ambos modelos en memoria simultáneamente. Como referencia de costos en AWS (región us-east-1, Linux, bajo demanda): una instancia t4g.large (8GB RAM, sin GPU) cuesta USD 0.0672/hora (menos de USD 12/mes operando 8 horas diarias en días laborales). Para reducir la latencia del LLM a ~0.3s, una instancia g4dn.xlarge (16GB RAM, GPU T4) cuesta USD 0.526/hora, aproximadamente USD 90/mes en las mismas condiciones.

NOTA: El costo fijo de Generación de políticas escala en función de la cantidad de documentos asociados.

---

## Riesgos y limitaciones

**Dataset pequeño:** el clasificador fue entrenado con 10 ejemplos. En producción se necesitarían al menos 50 ejemplos por clase para resultados robustos.

**LLM 3b no es determinista:** aunque temperatura=0, el modelo puede variar en casos donde el contexto es insuficiente o ambiguo. Se observó esto en el ticket "cargo delayed, may lose supermarket slot" que en distintas ejecuciones clasificó como `shipment_exception` o `temperature_exception`.

**Idioma:** el clasificador de embeddings usa `all-MiniLM-L6-v2`, entrenado principalmente en inglés. Tickets en otros idiomas tendrán menor calidad de clasificación. De todas formas existe soporte multilingüe, pero no fue revisado en esta demo.

**Politicas generadas por LLM 3b:** el modelo puede generar instrucciones incompletas. Por eso la validación humana del JSON es obligatoria antes de producción.

---

## Que mejoraría con una semana mas

- Agregar mas datos de train, al menos 50 ejemplos por clase para mejorar accuracy del clasificador
- Reemplazar Ollama por un modelo mas capaz para casos ambiguos
- Agregar evaluación automática del test set con labels manuales
- Exponer el agente como API REST con FastAPI
- Agregar soporte multilingüe con modelo de embeddings multilingüe
- Mejorar `draft_response`, personalizar respuestas y permitir validación posterior de las mismas
