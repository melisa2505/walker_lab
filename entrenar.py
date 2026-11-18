"""PPO con currículo, evaluación fija, exportación y presupuesto de tiempo real."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agente.agent import Agent
from walker_lab.environment import WalkerEnv
from walker_lab.evaluation import Evaluator
from walker_lab.scenarios import generate_scenario

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("caminar", ["plano", "suave"], "normal"),
    ("relieve", ["suave", "pendientes"], "normal"),
    ("obstaculos", ["suave", "pendientes", "escalones", "bloques", "huecos", "mixto"], "normal"),
    ("maraton", ["mixto", "maraton"], "normal"),
    ("hardcore", ["mixto", "maraton"], "hardcore-plus"),
]
SUITE = [
    "practica/plano.json", "practica/suave.json", "practica/mixto.json",
    "validacion/maraton_01.json", "hardcore/maraton_01.json",
]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


class CurriculumEnv(gym.Wrapper):
    def __init__(self, seed):
        super().__init__(WalkerEnv(profile="plano"))
        self.stage = 0
        self.rng = np.random.default_rng(seed)

    def set_stage(self, stage):
        self.stage = stage

    def reset(self, **kwargs):
        _, profiles, difficulty = STAGES[self.stage]
        # Conservar ejemplos fáciles para reducir olvido catastrófico.
        profile = str(self.rng.choice(profiles)) if self.rng.random() > 0.15 else "suave"
        self.env.profile = profile
        self.env.difficulty = difficulty if profile != "suave" else "normal"
        return self.env.reset(**kwargs)


def export_policy(model, normalizer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = dict(model.policy.mlp_extractor.policy_net.state_dict())
    index = len(model.policy.mlp_extractor.policy_net)
    state[f"{index}.weight"] = model.policy.action_net.weight.detach()
    state[f"{index}.bias"] = model.policy.action_net.bias.detach()
    data = {
        "hidden_sizes": [256, 256],
        "state_dict": {k: v.detach().cpu() for k, v in state.items()},
        "obs_mean": torch.tensor(normalizer.obs_rms.mean, dtype=torch.float32),
        "obs_var": torch.tensor(normalizer.obs_rms.var, dtype=torch.float32),
        "epsilon": normalizer.epsilon, "clip_obs": normalizer.clip_obs,
        "timesteps": model.num_timesteps,
    }
    temp = folder / "policy.pt.tmp"
    torch.save(data, temp)
    temp.replace(folder / "policy.pt")


def evaluate_suite(agent, output, record=True):
    reports = []
    for relative in SUITE:
        scenario = ROOT / "escenarios" / relative
        label = relative.replace("/", "_").replace(".json", "")
        evaluator = Evaluator(scenario, record_dir=output / label if record else None)
        report = evaluator.evaluate(agent)
        report["track"] = relative
        reports.append(report)
    return reports


class OvernightCallback(BaseCallback):
    def __init__(self, args, run, state=None):
        super().__init__()
        self.args, self.run = args, run
        self.stage = (state or {}).get("stage", 0)
        self.best_score = (state or {}).get("best_score", -1.0)
        self.passes = (state or {}).get("passes", 0)
        self.started = time.monotonic()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.last_eval = 0
        self.last_status = 0
        self.stop_requested = False
        self.last_reports = []
        self.eval_count = 0

    def _on_training_start(self):
        self.training_env.env_method("set_stage", self.stage)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        self.status("entrenando")

    def status(self, phase):
        elapsed = time.monotonic() - self.started
        data = {
            "pid": os.getpid(), "phase": phase, "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_hours": elapsed / 3600, "budget_hours": self.args.hours,
            "timesteps": self.model.num_timesteps, "stage": self.stage,
            "stage_name": STAGES[self.stage][0], "best_score": self.best_score,
            "passes": self.passes, "seed": self.args.seed,
            "last_evaluation": self.last_reports,
        }
        write_json(self.run / "estado.json", data)
        rows = "\n".join(
            f"| {r['track']} | {r['distance']:.2f} / {r['length']:.0f} | {r['reason']} |"
            for r in self.last_reports
        )
        (self.run / "INFORME.md").write_text(
            f"# Entrenamiento nocturno\n\nEstado: **{phase}**. Actualizado: {data['updated_at']}.\n\n"
            f"- Inicio: {self.started_at}\n- Tiempo: {elapsed / 3600:.2f} / {self.args.hours:.1f} horas\n"
            f"- Interacciones acumuladas: {self.model.num_timesteps:,}\n"
            f"- Currículo: {self.stage + 1}/5, {STAGES[self.stage][0]}\n"
            f"- Mejor fracción media de recorrido: {max(0, self.best_score):.2%}\n"
            "- Algoritmo: PPO, actor/crítico MLP 256×256, 4 entornos, CPU.\n"
            "- Observaciones y retornos normalizados durante entrenamiento; inferencia determinista.\n"
            "- El agente entregado corresponde al mejor promedio de distancia normalizada de la suite fija.\n"
            "- La suite usada para seleccionar NO constituye una prueba independiente de generalización.\n\n"
            "## Última evaluación del candidato\n\n| Pista | Distancia | Final |\n|---|---:|---|\n"
            f"{rows}\n\n"
            "## Archivos y uso\n\n"
            "- `agente/policy.pt`: mejor política, incluye normalización.\n"
            "- `entrenamiento/nocturno/checkpoints/`: último PPO y normalizador para reanudar.\n"
            "- `entrenamiento/nocturno/historial.jsonl`: evaluaciones sucesivas.\n"
            "- `resultados/mejor/`: informe y trayectorias del mejor agente.\n"
            "- `resultados/final/`: inferencia al terminar y vídeo si se pudo generar.\n\n"
            "Desde la raíz: `source .venv/bin/activate`, luego `python evaluar.py --render` "
            "o `python evaluar.py --scenario escenarios/practica/plano.json --render`.\n\n"
            "Consulte `ENTRENAMIENTO.md` para monitorizar, detener, reanudar y reproducir.\n"
        )

    def checkpoint(self):
        dest = self.run / "checkpoints"
        dest.mkdir(exist_ok=True)
        self.model.save(dest / "latest.tmp.zip")
        self.training_env.save(dest / "vecnormalize.tmp.pkl")
        (dest / "latest.tmp.zip").replace(dest / "latest.zip")
        (dest / "vecnormalize.tmp.pkl").replace(dest / "vecnormalize.pkl")
        write_json(dest / "state.json", {"stage": self.stage, "best_score": self.best_score, "passes": self.passes})

    def assess_curriculum(self, agent):
        fractions = []
        _, profiles, difficulty = STAGES[self.stage]
        for i, profile in enumerate(profiles):
            scene = generate_scenario(profile, seed=80000 + i, difficulty=difficulty)
            env = WalkerEnv(scenario=scene)
            try:
                obs, info = env.reset()
                agent.reset()
                done = False
                while not done:
                    obs, _, terminated, truncated, info = env.step(agent.act(obs))
                    done = terminated or truncated
                fractions.append(info["distance"] / info["length"])
            finally:
                env.close()
        competence = float(np.mean(fractions))
        self.passes = self.passes + 1 if competence >= 0.75 else 0
        if self.passes >= 2 and self.stage < len(STAGES) - 1:
            self.stage += 1
            self.passes = 0
            self.training_env.env_method("set_stage", self.stage)
            print(f"CURRÍCULO -> {STAGES[self.stage][0]}", flush=True)
        return competence

    def evaluate(self):
        candidate = self.run / "candidato"
        export_policy(self.model, self.training_env, candidate)
        agent = Agent(candidate)
        reports = evaluate_suite(agent, self.run / "evaluacion_temporal", record=False)
        score = float(np.mean([r["distance"] / r["length"] for r in reports]))
        competence = self.assess_curriculum(agent)
        self.last_reports = reports
        if score > self.best_score:
            self.best_score = score
            export_policy(self.model, self.training_env, ROOT / "agente")
            best_reports = evaluate_suite(agent, ROOT / "resultados" / "mejor")
            write_json(ROOT / "resultados" / "mejor" / "evaluaciones.json", best_reports)
            self.model.save(self.run / "checkpoints" / "best.zip")
            self.training_env.save(self.run / "checkpoints" / "best_vecnormalize.pkl")
        entry = {"timesteps": self.model.num_timesteps, "score": score,
                 "competence": competence, "stage": self.stage, "reports": reports,
                 "time": datetime.now().isoformat(timespec="seconds")}
        with (self.run / "historial.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        self.logger.record("evaluation/mean_distance_fraction", score)
        self.logger.record("evaluation/curriculum_competence", competence)
        self.logger.record("evaluation/stage", self.stage)
        self.eval_count += 1
        self.checkpoint()
        self.status("entrenando")

    def _on_step(self):
        elapsed = time.monotonic() - self.started
        if self.stop_requested or elapsed >= self.args.hours * 3600:
            return False
        if time.monotonic() - self.last_status > 60:
            self.status("entrenando")
            self.last_status = time.monotonic()
        return True

    def _on_rollout_end(self):
        if self.model.num_timesteps - self.last_eval >= self.args.eval_every:
            self.evaluate()
            self.last_eval = self.model.num_timesteps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "entrenamiento" / "nocturno")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.hours <= 0 or args.envs < 1 or args.eval_every < 1:
        parser.error("hours, envs y eval-every deben ser positivos")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Ya existe un entrenamiento activo en este directorio.")
    (run / "checkpoints").mkdir(exist_ok=True)
    checkpoint = run / "checkpoints"
    if (checkpoint / "latest.zip").exists() and not args.resume:
        parser.error("Ya hay un checkpoint: use --resume o elija otro --run-dir.")
    env = DummyVecEnv([lambda i=i: Monitor(CurriculumEnv(args.seed + i)) for i in range(args.envs)])
    env.seed(args.seed)
    state = None
    if args.resume:
        env = VecNormalize.load(checkpoint / "vecnormalize.pkl", env)
        model = PPO.load(checkpoint / "latest.zip", env=env, device="cpu")
        state = json.loads((checkpoint / "state.json").read_text())
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, target_kl=0.03,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -0.5},
            tensorboard_log=str(run / "tensorboard"), seed=args.seed, device="cpu", verbose=1,
        )
    callback = OvernightCallback(args, run, state)
    write_json(run / "config.json", vars(args) | {"run_dir": str(run)})
    try:
        model.learn(total_timesteps=1_000_000_000, callback=callback, reset_num_timesteps=not args.resume, log_interval=5)
        callback.checkpoint()
        callback.evaluate()
        callback.status("evaluando_final")
        best_agent = Agent(ROOT / "agente")
        final_reports = evaluate_suite(best_agent, ROOT / "resultados" / "final")
        # Reservar estas pistas fuera de la selección del mejor checkpoint.
        for relative in ["validacion/maraton_02.json", "hardcore/maraton_02.json"]:
            report = Evaluator(ROOT / "escenarios" / relative).evaluate(best_agent)
            report["track"] = relative
            final_reports.append(report)
        write_json(ROOT / "resultados" / "final" / "evaluaciones.json", final_reports)
        final_rows = "\n".join(
            f"| {r['track']} | {r['distance']:.2f} / {r['length']:.0f} | {r['reason']} |"
            for r in final_reports
        )
        (ROOT / "resultados" / "final" / "RESUMEN.md").write_text(
            "# Inferencia del mejor agente al terminar\n\n"
            "Las pistas maraton_02 se reservan para esta comprobación final y no seleccionan checkpoints.\n\n"
            "| Pista | Distancia | Final |\n|---|---:|---|\n" + final_rows + "\n"
        )
        if not args.no_video:
            try:
                from crear_video import create_video
                trajectory = ROOT / "resultados/final/validacion_maraton_01/maraton_01.npz"
                create_video(trajectory, ROOT / "resultados/final/maraton.mp4")
            except Exception as error:
                (run / "video_error.txt").write_text(repr(error))
                print(f"Vídeo no generado: {error}", flush=True)
        callback.status("finalizado")
    except BaseException as error:
        callback.checkpoint()
        callback.status("error: " + repr(error))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
