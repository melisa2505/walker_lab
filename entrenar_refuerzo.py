"""Afinado PPO en habilidades, transferencia al hardcore 02 y consolidación."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
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
from entrenar import ROOT, SUITE, export_policy, write_json
from walker_lab.environment import WalkerEnv
from walker_lab.replay import TrajectoryRecorder
from walker_lab.scenarios import generate_scenario, load_scenario, save_scenario, scenario_digest
from walker_lab.targeted import SKILLS, generate_targeted

PHASES = ("A_habilidades", "B_hardcore02", "C_consolidacion")
WEIGHTS = {"normal": 0.25, "habilidades": 0.40, "hardcore": 0.35}


def build_suite(folder, heldout=False):
    """Semillas de selección y prueba final en rangos ajenos al entrenamiento."""
    tracks = []
    if not heldout:
        for relative in SUITE + ["validacion/maraton_02.json", "hardcore/maraton_02.json"]:
            tracks.append({"path": str(ROOT / "escenarios" / relative),
                           "label": relative.replace("/", "_").removesuffix(".json"),
                           "group": "hardcore" if relative.startswith("hardcore/") else "normal"})
    # 220 millones se usó en la prueba técnica; reservar un conjunto nuevo para la ejecución larga.
    base = 230_000_000 if heldout else 120_000_000
    for i, skill in enumerate(SKILLS):
        for level in (1, 2):
            label = f"{'reservada_' if heldout else ''}{skill}_nivel{level}"
            path = folder / ("reservadas" if heldout else "seleccion") / f"{label}.json"
            save_scenario(generate_targeted(skill, base + i * 10 + level, level), path)
            tracks.append({"path": str(path), "label": label, "group": "habilidades"})
    if heldout:
        for i in range(4):
            difficulty = "normal" if i < 2 else "hardcore-plus"
            label = f"maraton_reservada_{i}"
            path = folder / "reservadas" / f"{label}.json"
            save_scenario(generate_scenario("maraton", base + 100 + i, difficulty=difficulty), path)
            tracks.append({"path": str(path), "label": label,
                           "group": "normal" if i < 2 else "hardcore"})
    return tracks


class FocusEnv(gym.Wrapper):
    def __init__(self, seed):
        super().__init__(WalkerEnv())
        self.rng = np.random.default_rng(seed)
        self.phase, self.level = 0, 0
        self.hardcore = load_scenario(ROOT / "escenarios/hardcore/maraton_02.json")
        self.rehearsal = [load_scenario(ROOT / "escenarios" / relative) for relative in (
            "validacion/maraton_01.json", "validacion/maraton_02.json", "hardcore/maraton_01.json")]

    def configure(self, phase, level):
        self.phase, self.level = phase, level

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 1..99 999 999: nunca generar semillas de selección/reservadas.
        scene_seed = int(self.rng.integers(1, 100_000_000))
        draw = self.rng.random()
        if self.phase == 0:
            mode = "gaps" if draw < 0.4 else "stairs" if draw < 0.8 else "easy"
        elif self.phase == 1:
            mode = "hardcore02" if draw < 0.5 else "skill" if draw < 0.8 else "easy"
        else:
            mode = ("hardcore02" if draw < 0.2 else "hardcore" if draw < 0.4
                    else "skill" if draw < 0.7 else "easy")
        if mode in ("gaps", "stairs", "skill"):
            choices = SKILLS[:2] if mode == "gaps" else SKILLS[2:4] if mode == "stairs" else SKILLS
            skill = str(self.rng.choice(choices))
            # Conservar variantes de niveles anteriores al subir dificultad.
            level = self.level if self.rng.random() > 0.2 else max(0, self.level - 1)
            scene = generate_targeted(skill, scene_seed, level)
        elif mode == "hardcore02":
            scene = self.hardcore
        elif mode == "hardcore":
            scene = generate_scenario("maraton", scene_seed, difficulty="hardcore-plus")
        else:
            # La prueba de afinado mostró olvido de maratones: repasar éxitos previos.
            replay = self.rng.random()
            if replay < 0.1:
                scene = self.rehearsal[2]
            elif replay < 0.3:
                scene = self.rehearsal[int(self.rng.integers(0, 2))]
            else:
                scene = generate_scenario(str(self.rng.choice(["plano", "suave", "pendientes"])), scene_seed)
        self.env.fixed_scenario = scene
        # El terreno hardcore02 es fijo, pero variar la condición inicial evita una sola trayectoria.
        return self.env.reset(seed=scene_seed, options=options)


def evaluate(agent, tracks, output=None):
    reports = []
    for track in tracks:
        scene = load_scenario(track["path"])
        env = WalkerEnv(scenario=scene)
        try:
            obs, info = env.reset()
            agent.reset()
            recorder = TrajectoryRecorder(env, scene["simulation_seed"]) if output else None
            total_reward, delta, saturated, count = 0.0, 0.0, 0, 0
            previous = None
            done = False
            while not done:
                action = agent.act(obs)
                if previous is not None:
                    delta += float(np.abs(action - previous).mean())
                saturated += int(np.sum(np.abs(action) >= 0.99))
                previous = action.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                count += 1
                done = terminated or truncated
                if recorder:
                    recorder.capture(env)
            report = dict(info, track=track["label"], group=track["group"],
                          training_return=total_reward, scenario_sha256=scenario_digest(scene),
                          simulation_seed=scene["simulation_seed"],
                          action_delta_mean=delta / max(1, count - 1),
                          action_saturation_fraction=saturated / max(1, count * 4))
            if recorder:
                target = output / f"{track['label']}.npz"
                recorder.save(target)
                report["trajectory"] = str(target)
            reports.append(report)
            print(f"EVAL {track['label']}: {info['distance']:.2f}/{info['length']:.2f} {info['reason']}", flush=True)
        finally:
            env.close()
    return reports


def summarize(reports):
    groups = {}
    for group in WEIGHTS:
        rows = [r for r in reports if r["group"] == group]
        groups[group] = {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "distance_fraction": float(np.mean([r["distance"] / r["length"] for r in rows])),
        }
    return {
        "groups": groups,
        "success_score": sum(WEIGHTS[g] * groups[g]["success_rate"] for g in WEIGHTS),
        "distance_score": sum(WEIGHTS[g] * groups[g]["distance_fraction"] for g in WEIGHTS),
    }


class RefuerzoRun:
    def __init__(self, args, model, env, tracks, heldout, saved=None):
        self.args, self.model, self.env = args, model, env
        self.run = args.run_dir
        self.results = self.run / "resultados"
        self.tracks, self.heldout = tracks, heldout
        saved = saved or {}
        self.elapsed_before = saved.get("elapsed_seconds", 0)
        self.started = time.monotonic()
        self.started_at = saved.get("started_at", datetime.now().isoformat(timespec="seconds"))
        self.initial_steps = saved.get("initial_steps", model.num_timesteps)
        self.phase = saved.get("phase", 0)
        self.level = saved.get("level", 0)
        self.passes = saved.get("passes", 0)
        self.best_metrics = saved.get("best_metrics")
        self.best_snapshot = saved.get("best_snapshot", "origen")
        self.last_reports = saved.get("last_reports", [])
        self.baseline_reports = json.loads((self.run / "baseline.json").read_text()) if saved else []
        self.stop_requested = False
        self.last_eval = model.num_timesteps
        self.last_checkpoint = model.num_timesteps
        self.last_status = 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    def state(self):
        return {"elapsed_seconds": self.elapsed(), "started_at": self.started_at,
                "initial_steps": self.initial_steps, "phase": self.phase, "level": self.level,
                "passes": self.passes, "best_metrics": self.best_metrics,
                "best_snapshot": self.best_snapshot, "last_reports": self.last_reports}

    def status(self, status):
        data = self.state() | {
            "pid": os.getpid(), "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "phase_name": PHASES[self.phase], "budget_hours": self.args.hours,
            "elapsed_hours": self.elapsed() / 3600, "timesteps": self.model.num_timesteps,
            "new_timesteps": self.model.num_timesteps - self.initial_steps,
            "ent_coef": float(self.model.ent_coef),
            "policy_std_mean": float(self.model.policy.log_std.detach().exp().mean()),
        }
        write_json(self.run / "estado.json", data)
        rows = "\n".join(
            f"| {r['track']} | {r['distance']:.2f}/{r['length']:.2f} | {r['reason']} |"
            for r in self.last_reports
        )
        (self.run / "INFORME.md").write_text(
            f"# Refuerzo de habilidades\n\nEstado: **{status}**. Actualizado: {data['updated_at']}.\n\n"
            f"- Inicio: {self.started_at}\n- Tiempo: {data['elapsed_hours']:.2f}/{self.args.hours:.1f} horas\n"
            f"- Fase: {PHASES[self.phase]}; dificultad de habilidades: {self.level}/2\n"
            f"- Pasos adicionales: {data['new_timesteps']:,}; acumulados: {data['timesteps']:,}\n"
            f"- Incentivo de entropía: {data['ent_coef']:.6f}; desviación media: {data['policy_std_mean']:.4f}\n"
            f"- Mejor selección: {json.dumps(self.best_metrics, ensure_ascii=False)}\n\n"
            "## Último candidato evaluado\n\n| Pista | Distancia | Final |\n|---|---:|---|\n"
            f"{rows}\n\n"
            "## Dónde ver los resultados\n\n"
            "- `entrenamiento/refuerzo/baseline.json`: agente anterior en la nueva suite.\n"
            "- `entrenamiento/refuerzo/resultados/mejor/`: mejor agente seleccionado y trayectorias.\n"
            "- `entrenamiento/refuerzo/resultados/final/COMPARACION.md`: antes/después al terminar.\n"
            "- `entrenamiento/refuerzo/resultados/final/videos/`: vídeos al terminar.\n"
            "- `REFUERZO.md`: explicación y comandos.\n\n"
            "Hardcore 02 ahora forma parte del entrenamiento. Las nuevas pistas reservadas se evalúan solo al final.\n"
        )

    def controls(self):
        fraction = self.elapsed() / (self.args.hours * 3600)
        phase = 0 if fraction < 4 / 12 else 1 if fraction < 9 / 12 else 2
        if phase != self.phase:
            self.phase = phase
            print(f"FASE -> {PHASES[phase]}", flush=True)
        # 0.001 durante A/B, descenso gradual a 0.0001 durante C.
        self.model.ent_coef = 0.001 if phase < 2 else 0.001 - 0.0009 * min(1, max(0, (fraction - 0.75) / 0.25))
        self.env.env_method("configure", self.phase, self.level)

    def snapshot(self, prefix):
        name = f"checkpoints/{prefix}_{self.model.num_timesteps}_{time.time_ns()}"
        dest = self.run / name
        dest.mkdir(parents=True)
        self.model.save(dest / "model.zip")
        self.env.save(dest / "normalizer.pkl")
        export_policy(self.model, self.env, dest)
        write_json(dest / "state.json", self.state())
        return name

    def checkpoint(self):
        name = self.snapshot("latest")
        # El puntero se publica solo después de guardar el conjunto completo.
        write_json(self.run / "latest.json", {"directory": name})
        self.last_checkpoint = self.model.num_timesteps
        snapshots = sorted((self.run / "checkpoints").glob("latest_*"), key=lambda p: p.stat().st_mtime)
        for old in snapshots[:-2]:
            shutil.rmtree(old)

    def publish(self, snapshot):
        source = self.run / snapshot / "policy.pt"
        for folder in (self.run / "mejor", self.args.publish_dir):
            folder.mkdir(parents=True, exist_ok=True)
            temp = folder / "policy.pt.tmp"
            shutil.copy2(source, temp)
            temp.replace(folder / "policy.pt")
        write_json(self.run / "best.json", {"directory": snapshot, "metrics": self.best_metrics})

    def initialize_baseline(self):
        self.status("evaluando_referencia")
        agent = Agent(self.run / "origen")
        self.baseline_reports = evaluate(agent, self.tracks, self.results / "anterior")
        write_json(self.run / "baseline.json", self.baseline_reports)
        self.best_metrics = summarize(self.baseline_reports)
        self.last_reports = self.baseline_reports
        self.publish("origen")
        write_json(self.results / "mejor/evaluaciones.json", self.baseline_reports)
        # El presupuesto de afinado empieza después de medir el agente de referencia.
        self.started = time.monotonic()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.checkpoint()

    def competence(self, agent):
        tracks = []
        for i, skill in enumerate(SKILLS):
            path = self.run / "pistas/competencia" / f"{skill}.json"
            save_scenario(generate_targeted(skill, 140_000_000 + self.level * 100 + i, self.level), path)
            tracks.append({"path": str(path), "label": f"competencia_{skill}", "group": "habilidades"})
        rows = evaluate(agent, tracks)
        success = float(np.mean([r["success"] for r in rows]))
        distance = float(np.mean([r["distance"] / r["length"] for r in rows]))
        self.passes = self.passes + 1 if success >= 0.6 and distance >= 0.8 else 0
        if self.passes >= 2 and self.level < 2:
            self.level += 1
            self.passes = 0
            print(f"DIFICULTAD -> {self.level}", flush=True)
        return {"success_rate": success, "distance_fraction": distance}

    def assess(self):
        export_policy(self.model, self.env, self.run / "candidato")
        agent = Agent(self.run / "candidato")
        reports = evaluate(agent, self.tracks)
        metrics = summarize(reports)
        competence = self.competence(agent)
        # Conservar los éxitos de la suite original antes de publicar un reemplazo.
        protected = {r["track"] for r in self.baseline_reports
                     if r["success"] and r["group"] != "habilidades"}
        preserves = all(r["success"] for r in reports if r["track"] in protected)
        key = lambda m: (round(m["success_score"], 10), m["distance_score"])
        improved = preserves and key(metrics) > key(self.best_metrics)
        if improved:
            self.best_metrics = metrics
            name = self.snapshot("best")
            self.best_snapshot = name
            self.publish(name)
            recorded = evaluate(agent, self.tracks, self.results / "mejor")
            write_json(self.results / "mejor/evaluaciones.json", recorded)
        self.last_reports = reports
        entry = {"time": datetime.now().isoformat(timespec="seconds"),
                 "elapsed_hours": self.elapsed() / 3600, "timesteps": self.model.num_timesteps,
                 "phase": self.phase, "level": self.level, "metrics": metrics,
                 "competence": competence, "published": improved, "preserves_baseline": preserves,
                 "ent_coef": self.model.ent_coef,
                 "policy_std_mean": float(self.model.policy.log_std.detach().exp().mean()), "reports": reports}
        with (self.run / "historial.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self.model.logger.record("evaluation/success_score", metrics["success_score"])
        self.model.logger.record("evaluation/distance_score", metrics["distance_score"])
        self.model.logger.record("evaluation/competence", competence["distance_fraction"])
        self.model.logger.record("curriculum/phase", self.phase)
        self.model.logger.record("curriculum/level", self.level)
        self.model.logger.record("exploration/ent_coef", self.model.ent_coef)
        self.last_eval = self.model.num_timesteps
        self.checkpoint()
        self.status("entrenando")

    def finish(self):
        self.status("evaluando_final")
        agent = Agent(self.run / "mejor")
        selected = evaluate(agent, self.tracks, self.results / "final")
        original = Agent(self.run / "origen")
        before_heldout = evaluate(original, self.heldout)
        after_heldout = evaluate(agent, self.heldout, self.results / "final/reservadas")
        write_json(self.results / "final/evaluaciones.json", selected)
        write_json(self.results / "final/reservadas_antes.json", before_heldout)
        write_json(self.results / "final/reservadas_despues.json", after_heldout)
        before = {r["track"]: r for r in self.baseline_reports + before_heldout}
        table = []
        for r in selected + after_heldout:
            old = before[r["track"]]
            table.append(f"| {r['track']} | {old['distance'] / old['length']:.1%} ({old['reason']}) "
                         f"| {r['distance'] / r['length']:.1%} ({r['reason']}) |")
        (self.results / "final/COMPARACION.md").write_text(
            "# Antes y después del refuerzo\n\n"
            "El agente final es el mejor seleccionado, que puede ser el original si ningún candidato lo supera.\n"
            "Las pistas reservadas se ejecutaron al final y no eligieron el modelo.\n\n"
            "## Selección y, después, pistas reservadas\n\n"
            "| Pista | Anterior | Seleccionado |\n|---|---:|---:|\n" + "\n".join(table) + "\n"
        )
        if not self.args.no_video:
            from crear_video import create_video
            for label in ("huecos_triples_nivel2", "combinado_nivel2", "hardcore_maraton_02"):
                try:
                    create_video(self.results / "final" / f"{label}.npz",
                                 self.results / "final/videos" / f"{label}.mp4")
                except Exception as error:
                    with (self.run / "video_errors.txt").open("a") as stream:
                        stream.write(f"{label}: {error!r}\n")
        self.status("detenido" if self.stop_requested else "finalizado")


class RefuerzoCallback(BaseCallback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def _on_training_start(self):
        self.experiment.controls()
        self.experiment.status("entrenando")

    def _on_step(self):
        run = self.experiment
        if run.stop_requested or run.elapsed() >= run.args.hours * 3600:
            return False
        if time.monotonic() - run.last_status > 60:
            run.controls()
            run.status("entrenando")
            run.last_status = time.monotonic()
        return True

    def _on_rollout_start(self):
        self.experiment.controls()

    def _on_rollout_end(self):
        run = self.experiment
        if self.model.num_timesteps - run.last_eval >= run.args.eval_every:
            run.assess()
        elif self.model.num_timesteps - run.last_checkpoint >= run.args.checkpoint_every:
            run.checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--seed", type=int, default=73117)
    parser.add_argument("--eval-every", type=int, default=200000)
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "entrenamiento/refuerzo")
    parser.add_argument("--publish-dir", type=Path, default=ROOT / "agente")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "entrenamiento/nocturno/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.hours <= 0 or args.eval_every < 1 or args.checkpoint_every < 1:
        parser.error("Presupuesto e intervalos deben ser positivos.")
    args.run_dir = args.run_dir.resolve()
    args.publish_dir = args.publish_dir.resolve()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Ya hay un entrenamiento activo en este directorio.")
    if (run / "latest.json").exists() and not args.resume:
        parser.error("Ya existe un checkpoint; use --resume o un run-dir nuevo.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved = None
    if args.resume:
        directory = json.loads((run / "latest.json").read_text())["directory"]
        source = run / directory
        saved = json.loads((source / "state.json").read_text())
    else:
        source = run / "origen"
        source.mkdir(exist_ok=True)
        shutil.copy2(args.source_dir / "best.zip", source / "model.zip")
        shutil.copy2(args.source_dir / "best_vecnormalize.pkl", source / "normalizer.pkl")
        shutil.copy2(ROOT / "agente/policy.pt", source / "policy_anterior_publicada.pt")
    tracks = build_suite(run / "pistas")
    heldout = build_suite(run / "pistas", heldout=True)
    write_json(run / "suite_seleccion.json", tracks)
    write_json(run / "suite_reservada.json", heldout)
    write_json(run / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    env = DummyVecEnv([lambda i=i: Monitor(FocusEnv(args.seed + i)) for i in range(4)])
    env.seed(args.seed)
    env = VecNormalize.load(source / "normalizer.pkl", env)
    env.training, env.norm_reward = True, True
    model = PPO.load(source / "model.zip", env=env, device="cpu")
    model.tensorboard_log = str(run / "tensorboard")
    # Afinado conservador; conservar pesos, optimizador y log_std aprendidos.
    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4
    model.ent_coef = 0.001
    if not args.resume:
        export_policy(model, env, source)
    experiment = RefuerzoRun(args, model, env, tracks, heldout, saved)
    try:
        if not args.resume:
            experiment.initialize_baseline()
        model.learn(total_timesteps=1_000_000_000, callback=RefuerzoCallback(experiment),
                    reset_num_timesteps=False, log_interval=5)
        experiment.checkpoint()
        experiment.assess()
        experiment.finish()
    except BaseException as error:
        experiment.checkpoint()
        experiment.status("error: " + repr(error))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
