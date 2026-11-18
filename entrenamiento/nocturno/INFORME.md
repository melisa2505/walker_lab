# Entrenamiento nocturno

Estado: **finalizado**. Actualizado: 2026-11-17T12:24:30.

- Inicio: 2026-11-17T02:23:48
- Tiempo: 10.01 / 10.0 horas
- Interacciones acumuladas: 42,855,344
- Currículo: 5/5, hardcore
- Mejor fracción media de recorrido: 100.00%
- Algoritmo: PPO, actor/crítico MLP 256×256, 4 entornos, CPU.
- Observaciones y retornos normalizados durante entrenamiento; inferencia determinista.
- El agente entregado corresponde al mejor promedio de distancia normalizada de la suite fija.
- La suite usada para seleccionar NO constituye una prueba independiente de generalización.

## Última evaluación del candidato

| Pista | Distancia | Final |
|---|---:|---|
| practica/plano.json | 60.00 / 60 | success |
| practica/suave.json | 60.00 / 60 | success |
| practica/mixto.json | 60.00 / 60 | success |
| validacion/maraton_01.json | 153.22 / 240 | fall |
| hardcore/maraton_01.json | 220.76 / 240 | fall |

## Archivos y uso

- `agente/policy.pt`: mejor política, incluye normalización.
- `entrenamiento/nocturno/checkpoints/`: último PPO y normalizador para reanudar.
- `entrenamiento/nocturno/historial.jsonl`: evaluaciones sucesivas.
- `resultados/mejor/`: informe y trayectorias del mejor agente.
- `resultados/final/`: inferencia al terminar y vídeo si se pudo generar.

Desde la raíz: `source .venv/bin/activate`, luego `python evaluar.py --render` o `python evaluar.py --scenario escenarios/practica/plano.json --render`.

Consulte `ENTRENAMIENTO.md` para monitorizar, detener, reanudar y reproducir.
