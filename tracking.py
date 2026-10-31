"""Compara las predicciones con los resultados de los partidos ya jugados de la temporada actual.

Cada predicción usa solo lo que se sabía antes del partido y, de los que jugaron, los 5 con más
minutos esperados (como si se hubiera revisado el reporte de lesiones), igual que en la evaluación
de model.py y points.py.
"""
import joblib
import numpy as np
import pandas as pd

from data import CURRENT_SEASON
from model import MODEL_PATH, build_dataset
from points import season_projections


def season_tracking(season, model):
    """Una fila por partido jugado: probabilidad del local, puntos proyectados y resultado real."""
    X, _ = build_dataset(season)
    games, projections = season_projections(season)
    ready = X.notna().all(axis=1)  # partidos cuyo box score ya está disponible
    team_home, team_away = projections["Por equipo"]
    lineup_home, lineup_away = projections["Por alineación"]
    track = games.assign(
        P_HOME=pd.Series(model.predict_proba(X[ready])[:, 1], index=X.index[ready]),
        TEAM_HOME=team_home, TEAM_AWAY=team_away, LINEUP_HOME=lineup_home, LINEUP_AWAY=lineup_away,
    )
    return track[ready].sort_values(["GAME_DATE", "GAME_ID"], ascending=False).reset_index(drop=True)


def projected_margin(track, method):
    """Diferencia del marcador proyectado: el equipo que queda arriba, por cuánto, y su margen real.

    El margen real es positivo si ese equipo ganó y negativo si perdió.
    """
    margin = track[f"{method}_HOME"] - track[f"{method}_AWAY"]
    favorite = pd.Series(np.where(margin >= 0, track["HOME_TEAM"], track["AWAY_TEAM"]), index=track.index)
    real = np.sign(margin) * (track["HOME_PTS"] - track["AWAY_PTS"])
    return favorite, margin.abs(), real


def summary(track):
    """Aciertos del ganador, error medio de los puntos (por equipo y del total) y diferencias cumplidas de cada método.

    La diferencia se cumple cuando el equipo que el marcador proyectado pone arriba gana por al menos esa diferencia.
    """
    hits = (track["P_HOME"] >= 0.5) == (track["HOME_WIN"] == 1)
    result = {"Partidos": len(track), "Aciertos del ganador": hits.mean()}
    for method in ["TEAM", "LINEUP"]:
        home_err = track[f"{method}_HOME"] - track["HOME_PTS"]
        away_err = track[f"{method}_AWAY"] - track["AWAY_PTS"]
        result[f"{method} por equipo"] = pd.concat([home_err, away_err]).abs().mean()
        result[f"{method} total"] = (home_err + away_err).abs().mean()
        _, margin, real = projected_margin(track, method)
        result[f"{method} diferencia"] = (real >= margin).mean()
    return result


def main():
    track = season_tracking(CURRENT_SEASON, joblib.load(MODEL_PATH))
    if track.empty:
        print(f"Todavía no hay partidos terminados de {CURRENT_SEASON}.")
        return
    s = summary(track)
    print(f"{CURRENT_SEASON}: {s['Partidos']} partidos | ganador acertado {s['Aciertos del ganador']:.1%}")
    print(f"Error de puntos por equipo: por equipo {s['TEAM por equipo']:.1f}, por alineación {s['LINEUP por equipo']:.1f}")
    print(f"Error del total: por equipo {s['TEAM total']:.1f}, por alineación {s['LINEUP total']:.1f}")
    print(f"Diferencia cumplida: por equipo {s['TEAM diferencia']:.1%}, por alineación {s['LINEUP diferencia']:.1%}")


if __name__ == "__main__":
    main()
