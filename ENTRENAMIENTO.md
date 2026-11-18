# Entrenamiento e inferencia

## Qué se preparó

Se creó `.venv/` con Python 3.12 y las versiones originales del simulador,
PyTorch 2.7.1 para CPU, Stable-Baselines3 2.7.1, TensorBoard y exportación MP4.
Las dependencias están en `requirements-entrenamiento.txt`; la inferencia del
agente requiere solamente el simulador, NumPy y PyTorch.

`entrenar.py` aprende **desde cero** mediante PPO: no usa pesos descargados.
La política recibe las 24 observaciones y produce las cuatro acciones del contrato.
Actor y crítico tienen dos capas de 256 unidades con Tanh. Se usan cuatro entornos
en un solo proceso para limitar RAM, un hilo de PyTorch, rollouts de 4096
transiciones, minibatches de 256, 10 épocas, learning rate 0.0003, gamma 0.99,
GAE 0.95, clipping 0.2 y KL objetivo 0.03. No se modificó la recompensa del laboratorio.

Se normalizan observaciones y retornos durante el entrenamiento. `agente/policy.pt`
incluye los pesos del actor **y** las estadísticas necesarias para inferencia.
La inferencia es determinista y no depende de Stable-Baselines3.

### Currículo

1. Plano y suave.
2. Suave y pendientes.
3. Mezcla de habilidades y obstáculos normales.
4. Mixto y maratón normal.
5. Mixto y maratón hardcore-plus.

Cada episodio genera terreno nuevo. Se conserva un 15 % de ejemplos suaves.
Para avanzar se requiere al menos 75 % de distancia media en pistas procedurales
fijas del nivel durante dos evaluaciones consecutivas. No se fuerza subir de nivel
sin haber aprendido: el nivel alcanzado dependerá de la ejecución.

Cada aproximadamente 100 000 interacciones se evalúan plano, suave, mixto,
maratón normal 01 y hardcore 01. Se selecciona el checkpoint con mayor
**media de distancia/longitud**, no con mayor recompensa. Estas pistas son
de selección; maratón normal 02 y hardcore 02 quedan para comprobación final.
Un mejor promedio no significa completar todas las pistas.

## Ejecución nocturna

Se configuró un presupuesto de **10 horas de tiempo real**, más el tiempo necesario
para la evaluación/exportación final. El proceso es independiente de la terminal,
tiene prioridad reducida e inhibe la suspensión ordinaria mientras está activo.
Mantén la computadora encendida, conectada a corriente y la sesión abierta;
deja la tapa abierta. Bloquear la pantalla está bien. No hace falta mantener
OpenCode abierto. Apagar o reiniciar interrumpe el entrenamiento.

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
python nocturno.py estado
tail -f entrenamiento/nocturno/salida.log
```

El estado se actualiza aproximadamente cada minuto, salvo durante evaluaciones.
Para salir de `tail`, usa Ctrl+C; eso no detiene el entrenamiento.

### Qué mirar al despertar

1. **`entrenamiento/nocturno/INFORME.md`**: informe actualizado automáticamente,
   estado, pasos, nivel y última evaluación del candidato.
2. **`resultados/mejor/evaluaciones.json`**: métricas del agente que se entrega.
3. **`resultados/final/RESUMEN.md`**: métricas finales, incluidas las dos pistas
   reservadas; aparece al finalizar el proceso.
4. **`resultados/final/maraton.mp4`**: vídeo de la inferencia final en la maratón 01.
5. **`entrenamiento/nocturno/historial.jsonl`**: evolución de todas las evaluaciones.

Se hizo una prueba técnica de aproximadamente un minuto antes de la ejecución
larga, guardada en `entrenamiento/prueba/`. Sus resultados iniciales no son evidencia
de que el robot ya camine. Los archivos de `resultados/final/` se reemplazan al
terminar la ejecución larga: comprueba `estado.json` para saber si ya finalizó.

### Ver al agente en directo

```bash
source .venv/bin/activate
python evaluar.py --scenario escenarios/practica/plano.json --render
python evaluar.py --render
python evaluar.py --scenario escenarios/hardcore/maraton_01.json --render
```

Estos comandos cargan el **mejor** agente publicado hasta ese momento, incluso
si el entrenamiento sigue activo. La política se reemplaza atómicamente al mejorar.

### Reproducir sin volver a ejecutar inferencia

```bash
python reproducir.py --trajectory resultados/mejor/practica_plano/plano.npz
python reproducir.py --trajectory resultados/mejor/validacion_maraton_01/maraton_01.npz
```

Crear un vídeo en cualquier momento:

```bash
python crear_video.py \
  --trajectory resultados/mejor/validacion_maraton_01/maraton_01.npz \
  --output resultados/maraton_actual.mp4
```

### Curvas con TensorBoard

```bash
.venv/bin/tensorboard --logdir entrenamiento/nocturno/tensorboard --host 127.0.0.1
```

Abre http://localhost:6006. Observa `rollout/ep_rew_mean`,
`evaluation/mean_distance_fraction`, `evaluation/curriculum_competence`
y `evaluation/stage`. El retorno de entrenamiento y la distancia no son equivalentes.

### Detener y continuar

```bash
python nocturno.py detener
# Espera a que termine la evaluación final; consulta:
python nocturno.py estado
# Continuar otras 10 horas desde el último checkpoint:
python nocturno.py iniciar --hours 10 --resume
```

La parada normal guarda el PPO completo y su normalizador y evalúa el mejor agente.
Si hay un corte abrupto, se puede recuperar el último checkpoint periódico en
`entrenamiento/nocturno/checkpoints/`. Se conserva también el PPO del mejor agente
en `best.zip` con `best_vecnormalize.pkl`. El último agente y el mejor pueden diferir.

## Verificaciones realizadas

- `comprobar_instalacion.py`: simulador operativo.
- Comprobador de API de Stable-Baselines3: entorno compatible.
- Validación de los 17 JSON de pistas.
- Entrenamiento corto completo, checkpoints, evaluaciones y vídeo MP4.
- Comparación de inferencia exportada contra PPO sobre 100 observaciones:
  diferencia máxima observada de aproximadamente 1.8e-7.

## Reinstalar dependencias

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-entrenamiento.txt \
  --index-strategy unsafe-best-match
```
