# Entrenamiento nocturno

Estado: **finalizado**. Actualizado: 2026-11-17T02:21:42.

- Inicio: 2026-11-17T02:20:30
- Tiempo: 0.02 / 0.0 horas
- Interacciones acumuladas: 40,964
- Currículo: 1/5, caminar
- Mejor fracción media de recorrido: 2.62%
- Algoritmo: PPO, actor/crítico MLP 256×256, 4 entornos, CPU.
- Observaciones y retornos normalizados durante entrenamiento; inferencia determinista.
- El agente entregado corresponde al mejor promedio de distancia normalizada de la suite fija.
- La suite usada para seleccionar NO constituye una prueba independiente de generalización.

## Última evaluación del candidato

| Pista | Distancia | Final |
|---|---:|---|
| practica/plano.json | 2.19 / 60 | fall |
| practica/suave.json | 2.29 / 60 | fall |
| practica/mixto.json | 2.30 / 60 | fall |
| validacion/maraton_01.json | 2.18 / 240 | fall |
| hardcore/maraton_01.json | 2.18 / 240 | fall |

## Archivos y uso

- `agente/policy.pt`: mejor política, incluye normalización.
- `entrenamiento/nocturno/checkpoints/`: último PPO y normalizador para reanudar.
- `entrenamiento/nocturno/historial.jsonl`: evaluaciones sucesivas.
- `resultados/mejor/`: informe y trayectorias del mejor agente.
- `resultados/final/`: inferencia al terminar y vídeo si se pudo generar.

Desde la raíz: `source .venv/bin/activate`, luego `python evaluar.py --render` o `python evaluar.py --scenario escenarios/practica/plano.json --render`.

Consulte `ENTRENAMIENTO.md` para monitorizar, detener, reanudar y reproducir.
