"""Iniciar, consultar y detener el entrenamiento independiente de la terminal."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "entrenamiento" / "nocturno"


def active_trainer(run=RUN, script="entrenar.py"):
    # Verificar el comando evita enviar señales a un PID reutilizado.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if (args and Path(os.fsdecode(args[0])).name.startswith("python")
                    and str(ROOT / script).encode() in args and str(run).encode() in args):
                return int(process.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["iniciar", "estado", "detener"])
    parser.add_argument("--hours", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refuerzo", action="store_true", help="Gestionar el nuevo entrenamiento de habilidades.")
    args = parser.parse_args()
    run = ROOT / "entrenamiento/refuerzo" if args.refuerzo else RUN
    script = "entrenar_refuerzo.py" if args.refuerzo else "entrenar.py"
    hours = args.hours if args.hours is not None else (12 if args.refuerzo else 10)
    pid = active_trainer(run, script)
    if args.command == "estado":
        print(f"Proceso activo: {pid}" if pid else "No hay un proceso de entrenamiento activo.")
        status = run / "estado.json"
        if status.exists():
            print(status.read_text())
        return
    if args.command == "detener":
        if pid:
            os.kill(pid, signal.SIGTERM)
            print(f"Se solicitó finalizar al proceso {pid}; guardará checkpoint y evaluará antes de salir.")
        else:
            print("No hay un entrenamiento activo.")
        return
    if pid:
        parser.error(f"Ya hay un entrenamiento activo: PID {pid}")
    if hours <= 0:
        parser.error("--hours debe ser positivo")
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "latest.json" if args.refuerzo else run / "checkpoints/latest.zip"
    if checkpoint.exists() and not args.resume:
        parser.error("Ya existe un checkpoint; use --resume para continuar.")
    command = [
        "nice", "-n", "10", "systemd-inhibit", "--what=idle:sleep", "--mode=block",
        "--who=WalkerLab", "--why=Entrenamiento nocturno solicitado",
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script),
        "--hours", str(hours), "--run-dir", str(run),
    ]
    if args.resume:
        command.append("--resume")
    environment = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                               "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    with (run / "salida.log").open("a") as log:
        log.write(f"\nInicio: {datetime.now().isoformat()}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, close_fds=True)
    (run / "proceso.json").write_text(json.dumps({"launcher_pid": process.pid, "command": command}, indent=2))
    print(f"Entrenamiento lanzado en segundo plano con presupuesto de {hours} horas. PID lanzador: {process.pid}")
    print(f"Registro: {run / 'salida.log'}")


if __name__ == "__main__":
    main()
