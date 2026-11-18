"""Convertir una trayectoria a MP4 sin ventana ni agente."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio.v2 as imageio

from walker_lab.environment import WalkerEnv
from walker_lab.replay import load_trajectory


def create_video(trajectory, output, stride=3):
    scene, poses, distances, steps = load_trajectory(trajectory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = WalkerEnv(scenario=scene, render_mode="rgb_array")
    try:
        env.reset()
        with imageio.get_writer(str(output), fps=50 / stride, codec="libx264", macro_block_size=2) as writer:
            for i in range(0, len(poses), stride):
                for body, (x, y, angle) in zip(env.legs + [env.hull], poses[i]):
                    body.position = (float(x), float(y))
                    body.angle = float(angle)
                env.max_distance = float(distances[i])
                env.steps = int(steps[i])
                writer.append_data(env.render())
    finally:
        env.close()
    print(f"Vídeo guardado: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("resultados/video.mp4"))
    args = parser.parse_args()
    create_video(args.trajectory, args.output)
