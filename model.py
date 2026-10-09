"""Entrena y evalúa la regresión logística que predice si gana el equipo local.

Cada partido se describe con las stats de la temporada ANTERIOR de los 5 jugadores
con más minutos de cada equipo, igual que en la app (2026-27 con stats de 2025-26).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import STAT_COLS, load, prev_season

MODEL_PATH = Path(__file__).parent / "model.joblib"
BENCH_MIN = 15  # jugadores con menos minutos por partido = perfil de banca


def stats_table(season):
    """Stats por jugador de una temporada y el perfil de banca para quien no tiene stats."""
    ps = load("player_stats", season).set_index("PLAYER_ID")[STAT_COLS]
    bench = ps[ps["MIN"] < BENCH_MIN].mean()
    return ps, bench


def game_features(home_ids, away_ids, ps, bench):
    home = ps.reindex(home_ids).fillna(bench).mean()
    away = ps.reindex(away_ids).fillna(bench).mean()
    return pd.concat([home.add_prefix("HOME_"), away.add_prefix("AWAY_"), (home - away).add_prefix("DIFF_")])


def contributions(model, features):
    """Aporte de cada stat al puntaje del local: positivo favorece al local, negativo al visitante."""
    scaler, logreg = model[0], model[-1]
    parts = (features - scaler.mean_) / scaler.scale_ * logreg.coef_[0]
    return pd.Series({s: parts[[f"HOME_{s}", f"AWAY_{s}", f"DIFF_{s}"]].sum() for s in STAT_COLS})


def build_dataset(season):
    games = load("games", season)
    ps, bench = stats_table(prev_season(season))
    lineups = (
        load("player_games", season)
        .sort_values("MIN", ascending=False)
        .groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .apply(lambda ids: list(ids[:5]))
    )
    X = pd.DataFrame([
        game_features(lineups[g.GAME_ID, g.HOME_TEAM_ID], lineups[g.GAME_ID, g.AWAY_TEAM_ID], ps, bench)
        for g in games.itertuples()
    ])
    return X, games["HOME_WIN"]


def main():
    X_train, y_train = build_dataset("2024-25")
    X_test, y_test = build_dataset("2025-26")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    baseline = y_test.mean()  # "siempre gana el local"
    print(f"Entrenamiento: {len(X_train)} partidos de 2024-25 (stats 2023-24)")
    print(f"Prueba:        {len(X_test)} partidos de 2025-26 (stats 2024-25)")
    print(f"Aciertos del modelo:       {accuracy:.1%}")
    print(f"Baseline (gana el local):  {baseline:.1%}")
    print(f"Mejora sobre el baseline:  {accuracy - baseline:+.1%}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
