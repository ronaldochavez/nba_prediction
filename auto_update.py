"""Actualiza los datos y los sube a GitHub sin intervención, para que la app en Streamlit Cloud los tenga.

stats.nba.com bloquea los servidores de la nube, así que la descarga tiene que hacerse desde una conexión de
casa. La tarea programada de Windows (programar_actualizacion.ps1) corre este script cada 5 minutos, pero
solo descarga cuando toca: 20 minutos antes de cada horario de partido, para tener la alineación confirmada,
y cada 3 horas. `python auto_update.py --ahora` actualiza en ese momento.
"""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data import CURRENT_SEASON, load, refresh_current

ROOT = Path(__file__).parent
STATE_PATH = ROOT / "auto_update.json"  # cuándo fue la última actualización; no se sube a GitHub
LOG_PATH = ROOT / "auto_update.log"
BEFORE_GAME = pd.Timedelta(minutes=20)
EVERY = pd.Timedelta(hours=3)
RETRY = pd.Timedelta(minutes=15)  # espera antes de reintentar una actualización que falló
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # que git no abra una consola cada vez


def pregame_times():
    """Momentos de actualizar antes de los partidos: 20 minutos antes de cada horario de inicio."""
    schedule = load("schedule", CURRENT_SEASON)
    if "GAME_TIME_UTC" not in schedule:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(schedule["GAME_TIME_UTC"], utc=True).dropna().drop_duplicates() - BEFORE_GAME


def is_due(now, state):
    """Toca si desde la última actualización pasaron 3 horas o algún momento previo a un partido.

    Después de un intento fallido se esperan 15 minutos antes de volver a intentar.
    """
    last = pd.Timestamp(state["success"]) if state.get("success") else None
    attempt = pd.Timestamp(state["attempt"]) if state.get("attempt") else None
    if attempt is not None and (last is None or attempt > last) and now - attempt < RETRY:
        return False
    if last is None or now - last >= EVERY:
        return True
    times = pregame_times()
    return bool(((times > last) & (times <= now)).any())


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, creationflags=NO_WINDOW
    ).stdout


def publish():
    """Hace commit solo de data/ si cambió y lo sube, después de traer lo que haya en GitHub."""
    git("add", "data")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", "data"], cwd=ROOT, creationflags=NO_WINDOW)
    if staged.returncode:
        git("commit", "-m", f"Actualizar datos automáticamente ({dt.datetime.now():%d/%m %H:%M})", "--", "data")
        print("commit hecho")
    try:
        # Los commits locales solo traen datos: si chocan con algo subido desde otro lado, ganan los recién bajados
        git("pull", "--rebase", "--autostash", "-X", "theirs")
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], cwd=ROOT, capture_output=True, creationflags=NO_WINDOW)
        raise
    git("push")
    print("subido a GitHub")


def main():
    if sys.stdout is None:  # con pythonw (la tarea programada) no hay consola: todo va al log
        sys.stdout = sys.stderr = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    now = pd.Timestamp.now(tz="UTC")
    if "--ahora" not in sys.argv and not is_due(now, state):
        return
    print(f"\n{dt.datetime.now():%d/%m %H:%M} actualizando")
    state["attempt"] = now.isoformat()
    try:
        refresh_current()
        state["success"] = now.isoformat()
    except Exception as e:  # lo que sí se descargó se sube de todos modos
        print(e)
    try:
        publish()
    except subprocess.CalledProcessError as e:
        print(f"git {' '.join(e.cmd[1:])}: {e.stderr.strip()}")
    STATE_PATH.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
