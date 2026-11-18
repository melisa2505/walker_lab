"""Guardar una variante de huecos consecutivos y escaleras grandes."""

import argparse

from walker_lab.scenarios import save_scenario
from walker_lab.targeted import SKILLS, generate_targeted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=SKILLS, default="combinado")
    parser.add_argument("--level", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--output", default="escenarios/refuerzo/combinado.json")
    args = parser.parse_args()
    scene = generate_targeted(args.skill, args.seed, args.level)
    save_scenario(scene, args.output)
    print(f"Guardado {args.output}: {scene['length']:.1f} unidades; {len(scene['events'])} eventos.")
