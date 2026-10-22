"""Entrena y evalúa la regresión logística que predice si gana el equipo local.

Cada partido se describe con la diferencia (local - visitante) del promedio de stats de la
temporada ANTERIOR de los 5 jugadores con más minutos de cada equipo, igual que en la app
(2026-27 con stats de 2025-26).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SEASONS, STAT_COLS, load, prev_season

MODEL_PATH = Path(__file__).parent / "model.joblib"
BENCH_MIN = 15  # jugadores con menos minutos por partido = perfil de banca
TEST_SEASONS = 3  # temporadas finales que se usan para evaluar


def stats_table(season):
    """Stats por jugador de una temporada y el perfil de banca para quien no tiene stats.

    Si un jugador no jugó esa temporada (p. ej. por una lesión), se usan sus stats de la anterior.
    """
    ps = load("player_stats", season).set_index("PLAYER_ID")[STAT_COLS]
    bench = ps[ps["MIN"] < BENCH_MIN].mean()
    older = load("player_stats", prev_season(season)).set_index("PLAYER_ID")[STAT_COLS]
    return ps.combine_first(older), bench


def lineup_stats(player_ids, ps, bench):
    """Promedio de stats de una alineación; los jugadores sin stats cuentan como banca."""
    return ps.reindex(player_ids).fillna(bench).mean()


def game_features(home_ids, away_ids, ps, bench):
    """Diferencia local - visitante del promedio de cada stat."""
    return lineup_stats(home_ids, ps, bench) - lineup_stats(away_ids, ps, bench)


def contributions(model, features):
    """Aporte de cada stat al puntaje del local: positivo favorece al local, negativo al visitante."""
    scaler, logreg = model[0], model[-1]
    return (features - scaler.mean_) / scaler.scale_ * logreg.coef_[0]


def weights(model):
    """Peso de cada stat (sobre stats estandarizadas) y su importancia relativa."""
    coef = pd.Series(model[-1].coef_[0], index=STAT_COLS)
    importance = coef.abs() / coef.abs().sum()
    return pd.DataFrame({"Peso": coef, "Importancia": importance}).sort_values("Importancia", ascending=False)


def game_lineups(season):
    """Los 5 jugadores con más minutos de cada equipo en cada partido, por (GAME_ID, TEAM_ID)."""
    return (
        load("player_games", season)
        .sort_values("MIN", ascending=False)
        .groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .apply(lambda ids: list(ids[:5]))
    )


def build_dataset(season):
    games = load("games", season)
    ps, bench = stats_table(prev_season(season))
    lineups = game_lineups(season)
    X = pd.DataFrame([
        game_features(lineups[g.GAME_ID, g.HOME_TEAM_ID], lineups[g.GAME_ID, g.AWAY_TEAM_ID], ps, bench)
        for g in games.itertuples()
    ])
    return X, games["HOME_WIN"]


def train(datasets):
    X = pd.concat([X for X, _ in datasets])
    y = pd.concat([y for _, y in datasets])
    # Las stats están muy correlacionadas (PTS y MIN: 0.90). La regularización, elegida por
    # validación cruzada, evita que el modelo reparta el peso entre ellas al azar.
    logreg = LogisticRegressionCV(Cs=20, l1_ratios=(0,), scoring="neg_log_loss", max_iter=2000, use_legacy_attributes=False)
    return make_pipeline(StandardScaler(), logreg).fit(X, y)


def main():
    datasets = {s: build_dataset(s) for s in SEASONS}

    print("Evaluación: cada temporada se predice con un modelo entrenado solo con las anteriores\n")
    print(f"{'Prueba':<9}{'Entrenamiento':<22}{'Modelo':>7}{'Siempre local':>15}")
    accuracies, baselines = [], []
    for i in range(len(SEASONS) - TEST_SEASONS, len(SEASONS)):
        model = train([datasets[s] for s in SEASONS[:i]])
        X, y = datasets[SEASONS[i]]
        accuracies.append(accuracy_score(y, model.predict(X)))
        baselines.append(y.mean())  # "siempre gana el local"
        print(f"{SEASONS[i]:<9}{SEASONS[0] + ' a ' + SEASONS[i - 1]:<22}{accuracies[-1]:>7.1%}{baselines[-1]:>15.1%}")
    print(f"{'Promedio':<31}{sum(accuracies) / len(accuracies):>7.1%}{sum(baselines) / len(baselines):>15.1%}")

    model = train(datasets.values())
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo final: {sum(len(y) for _, y in datasets.values())} partidos de {SEASONS[0]} a {SEASONS[-1]}")
    print(weights(model).to_string(formatters={"Peso": "{:+.2f}".format, "Importancia": "{:.0%}".format}))
    print(f"Ventaja de local: {model[-1].intercept_[0]:+.2f}")
    print(f"Guardado en {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
