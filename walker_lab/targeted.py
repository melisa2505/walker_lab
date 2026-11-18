"""Pistas de habilidades con geometría explícita y variantes reproducibles."""

import numpy as np
from gymnasium.envs.box2d.bipedal_walker import LEG_H

from .scenarios import BASE_Y, validate_scenario

SKILLS = ("huecos_dobles", "huecos_triples", "escaleras", "bajada", "combinado")


def generate_targeted(skill="combinado", seed=0, level=1):
    if skill not in SKILLS or level not in (0, 1, 2):
        raise ValueError("Habilidad desconocida o nivel fuera de 0..2.")
    rng = np.random.default_rng(seed)
    start = 14 / 3
    points = [[0.0, BASE_Y], [10.0, BASE_Y]]
    events = []
    cursor = float(rng.uniform(10.5, 13.0))

    def add(x, y):
        point = [round(float(x), 6), round(float(y), 6)]
        if point != points[-1]:
            points.append(point)

    def gaps(count):
        nonlocal cursor
        limits = [(0.75, 1.0), (1.0, 1.25), (1.18, 1.45)][level]
        separation = [(3.0, 3.8), (2.6, 3.5), (2.3, 3.3)][level]
        for _ in range(count):
            width = float(rng.uniform(*limits))
            x0 = cursor
            add(cursor, BASE_Y)
            add(cursor, BASE_Y - 2.0)
            cursor += width
            add(cursor, BASE_Y - 2.0)
            add(cursor, BASE_Y)
            events.append({"kind": "huecos", "x_start": x0, "x_end": cursor,
                           "width": width, "stage": level + 1})
            cursor += float(rng.uniform(*separation))
            add(cursor, BASE_Y)

    def stairs(descent_only=False):
        nonlocal cursor
        # LEG_H = 1.1333: dos segmentos dan una pierna de ~2.27 unidades.
        ratios = [(0.21, 0.26), (0.28, 0.35), (0.37, 0.44)][level]
        height = LEG_H * float(rng.uniform(*ratios))
        count = int(rng.integers(4 + level // 2, 6 + level // 2))
        width = float(rng.uniform(1.5, 1.95))
        x0 = cursor
        add(cursor, BASE_Y)
        if descent_only:
            # Rampa de acceso: permite practicar bajada sin superar antes la subida.
            cursor += max(8.0, count * height * 5)
            add(cursor, BASE_Y + count * height)
        else:
            for j in range(count):
                add(cursor, BASE_Y + (j + 1) * height)
                cursor += width
                add(cursor, BASE_Y + (j + 1) * height)
        cursor += float(rng.uniform(2.5, 3.5))
        add(cursor, BASE_Y + count * height)
        descent_start = cursor
        for j in reversed(range(count)):
            add(cursor, BASE_Y + j * height)
            cursor += width
            add(cursor, BASE_Y + j * height)
        events.append({"kind": "escalones", "x_start": x0, "x_end": cursor,
                       "height": height, "width": width, "count": count,
                       "peak_height": count * height, "descent_start": descent_start,
                       "descent_only": descent_only, "stage": level + 1})
        cursor += float(rng.uniform(3.5, 5.0))
        add(cursor, BASE_Y)

    if skill.startswith("huecos"):
        gaps(2 if skill == "huecos_dobles" else 3)
    elif skill in ("escaleras", "bajada"):
        stairs(descent_only=skill == "bajada")
    else:
        gaps(2)
        stairs()
        gaps(3)
    length = max(32.0, cursor - start + 8.0)
    finish = start + length
    add(finish + 10, BASE_Y)
    scene = {
        "profile": "mixto", "difficulty": "hardcore-plus", "seed": int(seed),
        "simulation_seed": int(seed), "start_x": start, "length": length,
        "finish_x": finish, "max_steps": max(1800, int(length * 55)),
        "ground": points, "obstacles": [], "events": events,
        "targeted_skill": skill, "targeted_level": level, "generator_version": 1,
        "robot_leg_segment_length": LEG_H,
    }
    validate_scenario(scene)
    return scene
