# Refuerzo: huecos consecutivos, escaleras y hardcore 02

## Objetivo y ejecución

Se preparó una nueva ejecución de **12 horas de afinado**, desde el checkpoint
`entrenamiento/nocturno/checkpoints/best.zip` y su normalizador correspondiente.
No parte del último modelo ni de pesos aleatorios. El presupuesto empieza después
de medir la referencia inicial y se añade evaluación/exportación al finalizar.

El original se conserva en `entrenamiento/refuerzo/origen/`, con el PPO completo,
normalizador, actor exportado y copia del actor publicado antes del refuerzo.
Los resultados de la ejecución anterior siguen en `resultados/final/`.

Los programas nuevos son:

- `walker_lab/targeted.py`: geometría de las familias de habilidades.
- `generar_refuerzo.py`: guardar una variante reproducible.
- `entrenar_refuerzo.py`: afinado PPO, evaluación, selección y exportación.
- `nocturno.py --refuerzo`: gestión del proceso separado de la terminal.

## Pistas nuevas

En `escenarios/refuerzo/` hay ejemplos fijos de cinco familias, cada una en niveles
0, 1 y 2. Durante el entrenamiento se generan variantes nuevas por episodio.

| Familia | Maniobra |
|---|---|
| `huecos_dobles` | Aterrizar y afrontar otro hueco |
| `huecos_triples` | Recuperar estabilidad entre tres huecos |
| `escaleras` | Subir, cruzar una plataforma y bajar |
| `bajada` | Rampa de acceso y escalera de bajada |
| `combinado` | Dos huecos, escalera de subida/bajada y tres huecos |

Todos tienen aproximación plana y salida estable, sin teletransportar al robot.
Los fosos tienen 2 unidades de profundidad. Sus anchos progresan de 0.75–1.0 a
1.18–1.45 unidades; las separaciones entre fosos pasan de 3.0–3.8 a 2.3–3.3.

Cada segmento de pierna mide 1.1333 unidades. Las contrahuellas se dimensionan
como una fracción de esa longitud: aproximadamente 0.24–0.29, 0.32–0.40 y
0.42–0.50 según el nivel. Hay 4–6 peldaños por subida/bajada, huellas de 1.5–1.95
y una plataforma superior de 2.5–3.5. Se obtiene una escalera alta mediante varios
peldaños, no un único salto desproporcionado.

### Visualizar

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
python visualizar.py --scenario escenarios/refuerzo/combinado_nivel2.json
python evaluar.py --scenario escenarios/refuerzo/combinado_nivel2.json --render
python evaluar.py --scenario escenarios/refuerzo/huecos_triples_nivel2.json --render
python evaluar.py --scenario escenarios/hardcore/maraton_02.json --render
```

`evaluar.py` usa el mejor agente publicado en `agente/policy.pt`. Al comenzar es
el agente original; solo se reemplaza cuando un candidato supera el criterio nuevo.

Generar una variante propia:

```bash
python generar_refuerzo.py --skill combinado --level 2 --seed 45678 \
  --output escenarios/refuerzo/mi_variante.json
```

## Currículo

### A: habilidades, horas 0–4

- 40 % episodios de huecos dobles/triples.
- 40 % episodios de escaleras o bajadas.
- 20 % mantenimiento: mayormente plano, suave o pendientes y un pequeño repaso
  de maratones ya superadas.

### B: transferencia, horas 4–9

- 50 % hardcore 02 completo, variando la condición inicial.
- 30 % habilidades, incluido combinado.
- 20 % mantenimiento.

### C: consolidación, horas 9–12

- 20 % hardcore 02.
- 20 % maratones hardcore-plus nuevas.
- 30 % habilidades y combinaciones.
- 30 % mantenimiento.

Dentro del bloque de mantenimiento se elige 70 % terreno fácil, 20 % maratones
normales 01/02 y 10 % hardcore 01. En A/B esto equivale a 14 % fáciles,
4 % maratones normales y 2 % hardcore 01 del total de episodios. Este pequeño
repaso se añadió después de observar olvido de maratones en la prueba corta.

Los cambios de fase se basan en el tiempo real acumulado. La **dificultad de las
habilidades** sube por competencia: al menos 60 % de éxito y 80 % de distancia
media en cinco pistas fijas del nivel, durante dos evaluaciones consecutivas.
Los niveles anteriores siguen apareciendo en el 20 % de los episodios de habilidad.
La dificultad no se fuerza al cambiar de fase si aún no se domina el nivel actual.

## PPO y exploración

Se mantienen arquitectura (actor/crítico MLP 256×256), recompensa, optimizador
cargado, estadísticas de normalización y desviación aprendida de las acciones.
Se reduce el learning rate de afinado a **0.0001**, para actualizar con más cautela.

Se añade un coeficiente de entropía pequeño: **0.001** durante A/B, que disminuye
linealmente hasta **0.0001** durante C. No se añade ruido externo a los motores ni
se reinicia `log_std`. Evaluación e inferencia siguen siendo deterministas.

Se registra la desviación de la política, diferencia media entre acciones
consecutivas y proporción de acciones saturadas. Son diagnósticos: no cambian
la recompensa. El incentivo de entropía no garantiza por sí solo suavidad.

## Selección del agente

Cada aproximadamente 200 000 interacciones se evalúan 17 pistas:

- Las cinco pistas de selección originales.
- Maratón normal 02 y hardcore 02.
- Dos variantes por habilidad, niveles 1 y 2.

Se agrupan en normal, habilidades y hardcore con pesos 25 %, 40 % y 35 %.
Se compara primero tasa de éxito ponderada y, en empate, distancia normalizada
ponderada. Además, el candidato debe conservar los éxitos del agente inicial
en las pistas normales/hardcore de esta suite. Un avance en habilidades no
publica un reemplazo que pierda esas pistas ya superadas.

Si no aparece una mejora admisible, el agente final permanece igual al original.
El último checkpoint puede ser peor que el mejor y se conserva para reanudar.

**Hardcore 02 ahora forma parte del entrenamiento.** Fue una reserva elegida
para el experimento anterior, no una restricción impuesta por el laboratorio.

Para la prueba final se guardan **14 pistas nuevas**: diez de habilidades y cuatro
maratones (dos normales, dos hardcore-plus). Se ejecutan con el agente anterior
y con el seleccionado solo al terminar; no determinan qué checkpoint se publica.
Las semillas se separan por rangos: entrenamiento por debajo de 100 millones,
selección desde 120 millones, competencia desde 140 millones y reserva desde
230 millones (el rango de 220 millones se usó solo en la prueba técnica).
Las pistas conocidas del proyecto se identifican por separado.

## Monitorización

```bash
source .venv/bin/activate
python nocturno.py estado --refuerzo
tail -f entrenamiento/refuerzo/salida.log
```

Archivos principales:

- `entrenamiento/refuerzo/INFORME.md`: estado y última evaluación del candidato.
- `entrenamiento/refuerzo/baseline.json`: rendimiento inicial en la suite nueva.
- `entrenamiento/refuerzo/resultados/mejor/evaluaciones.json`: mejor seleccionado.
- `entrenamiento/refuerzo/historial.jsonl`: historia de candidatos y publicaciones.
- `entrenamiento/refuerzo/resultados/final/COMPARACION.md`: antes/después al terminar.
- `entrenamiento/refuerzo/resultados/final/reservadas_*.json`: prueba independiente final.
- `entrenamiento/refuerzo/resultados/final/videos/`: MP4 de huecos triples,
  recorrido combinado y hardcore 02.

La primera evaluación de referencia tarda un poco; el estado indicará
`evaluando_referencia`. El registro muestra cada pista terminada.

TensorBoard:

```bash
.venv/bin/tensorboard --logdir entrenamiento/refuerzo/tensorboard --host 127.0.0.1
```

Abrir http://localhost:6006.

### Parar y reanudar

```bash
python nocturno.py detener --refuerzo
# Esperar a que finalicen la evaluación y los vídeos:
python nocturno.py estado --refuerzo
# Continuar el presupuesto original de 12 horas:
python nocturno.py iniciar --refuerzo --resume --hours 12
```

La reanudación cuenta el tiempo acumulado guardado: `--hours` es el **presupuesto
total**, no horas adicionales. Si ya completó 12 horas, para extenderlo hay que
indicar un total mayor; eso también reescala las fronteras temporales de fase.

Cada aproximadamente 50 000 interacciones se guarda un conjunto consistente de
PPO, normalizador, actor y estado. `latest.json` apunta al conjunto completo y se
actualiza atómicamente. Se retienen dos checkpoints periódicos y los mejores.

El proceso se lanza separado de la terminal, con prioridad reducida y bloqueo
de suspensión ordinaria. Deja la laptop conectada, encendida y con tapa abierta.
Si realmente se suspende, la CPU deja de entrenar durante ese intervalo.

## Prueba técnica

Antes de la ejecución larga se usa un directorio independiente:
`entrenamiento/prueba_refuerzo/`. Su agente se publica allí, no en `agente/`.
La prueba verifica carga del mejor PPO, entrenamiento, evaluaciones, guardado y
comparación final sin sustituir el agente de uso normal.
