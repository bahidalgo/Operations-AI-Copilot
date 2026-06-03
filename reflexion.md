# Reflexión técnica

**Autor:** Byron Hidalgo

---

## 1. ¿Cuál es la parte más frágil de la solución?

Hay dos puntos frágiles principales.

El primero es el clasificador de embeddings cuando opera con confianza alta pero predicción incorrecta, pues el umbral de confianza solo activa el respaldo LLM cuando la similitud es baja, pero se puede dar un caso donde un ticket puede tener similitud alta con el ejemplo equivocado y el sistema no lo detectaría. Una mejora concreta sería agregar una capa de reglas basadas en vocabulario crítico, por ejemplo, si el texto contiene palabras asociadas a casos de alta urgencia (por ejemplo, términos que el SLA matrix marca como respuesta en 30 minutos), forzar el paso por el LLM independientemente de la confianza del clasificador de embeddings.

El segundo es la revisión manual de políticas. Actualmente no existe un proceso formalizado de aprobación, por ahora se espera que un humano revise el JSON pero no hay versionado, ni rollback, ni trazabilidad de quién aprobó qué versión. En producción, una política incorrecta podría derivar casos al equipo equivocado sin que nadie lo detecte.

---

## 2. ¿Qué no lanzarías aún a producción?

No lanzaría `draft_response` sin un mecanismo de feedback del operador. Actualmente el sistema genera respuestas para el cliente y el operador sin que nadie pueda indicar si fueron útiles o no. Sin ese feedback loop no hay forma de mejorar el sistema ni de detectar patrones de error en producción. El operador debería poder marcar cada respuesta como útil o no, y esos datos deberían usarse para evaluar y mejorar el sistema con el tiempo.

---

## 3. ¿Qué construiste con ayuda de IA y cómo lo validaste?

Usé IA (Claude) para de desarrollo durante todo el proceso. Me ayudó a estructurar el código, redactar docstrings, y pensar decisiones de arquitectura.

La validación fue incremental y deliberada. Primero validé cada componente de forma independiente, por ejemplo, el clasificador de embeddings por sí solo, luego el LLM solo, las políticas solas, etc. Así podía asegurar test individuales antes de integrarlos en el agente. Esto me permitió detectar problemas en cada capa sin que el ruido de los otros componentes los escondieran.

Los tests automatizados cubrieron los contratos de cada función y los guardrails críticos. La evaluación manual sobre el test set me permitió observar el comportamiento real del sistema en casos ambiguos, incluyendo casos donde el LLM corregía al clasificador de embeddings y casos donde ambos fallaban.

Decidí un sistema interactivo para poder validar cómo funcionaba el sistema al considerar una mayor variabilidad en las querys, por ejemplo, es posible medir estabilidad frente a pequeños cambios (¿Qué pasa si cambio una sola palabra específica?).

Decidí no implementar RAG con chunking. Evalué si era necesario y concluí que con los pocos documentos vigentes de menos de 200 palabras cada uno, el chunking añadiría complejidad sin beneficio real. En producción con documentos más extensos o mayor volumen de políticas, sería la primera mejora a implementar.
