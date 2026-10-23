"""Entrena y evalúa la regresión logística que predice si gana el equipo local.

Cada partido se describe con la diferencia (local - visitante) del promedio de stats de los 5
jugadores con más minutos de cada equipo. Las stats de cada jugador mezclan su temporada anterior
con lo que llevaba de la actual antes de ese partido, igual que en la app.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SEASONS, STAT_COLS, load, prev_season

MODEL_PATH = Path(__file__).parent / "model.joblib"
BENCH_MIN = 15  # jugadores con menos minutos por partido = perfil de banca
TEST_SEASONS = 3  # temporadas finales que se usan para evaluar
# Partidos que "vale" la temporada anterior al mezclarla con la actual: a los 10 partidos jugados,
# mitad y mitad. Elegido probando 2, 5, 10, 20 y 40 en las temporadas 2023-24 a 2025-26.
K_GAMES = 10
SUM_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "PLUS_MINUS", "FGM", "FGA", "FG3M", "FG3A"]


def stats_table(season):
    """Stats por jugador de una temporada y el perfil de banca para quien no tiene stats.

    Si un jugador no jugó esa temporada (p. ej. por una lesión), se usan sus stats de la anterior.
    """
    ps = load("player_stats", season).set_index("PLAYER_ID")[STAT_COLS]
    bench = ps[ps["MIN"] < BENCH_MIN].mean()
    older = load("player_stats", prev_season(season)).set_index("PLAYER_ID")[STAT_COLS]
    return ps.combine_first(older), bench


def regular_season_logs(season):
    logs = load("game_logs", season)
    logs = logs[logs["SEASON_TYPE"] == "Regular Season"]
    return logs.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def per_game(totals, gp):
    """Promedios por partido a partir de totales; FG% y 3P% salen del total de tiros."""
    avg = totals[[c for c in STAT_COLS if not c.endswith("_PCT")]].div(gp.replace(0, np.nan), axis=0)
    avg["FG_PCT"] = totals["FGM"] / totals["FGA"].replace(0, np.nan)
    avg["FG3_PCT"] = totals["FG3M"] / totals["FG3A"].replace(0, np.nan)
    return avg[STAT_COLS]


def blend(prev, current, gp, k=K_GAMES):
    """Mezcla la temporada anterior con la actual; la actual pesa más con cada partido jugado."""
    w = gp / (gp + k)
    return prev.mul(1 - w, axis=0) + current.fillna(prev).mul(w, axis=0)


def current_stats(season):
    """Stats de cada jugador para predecir los próximos partidos de `season`, y el perfil de banca."""
    ps, bench = stats_table(prev_season(season))
    logs = regular_season_logs(season)
    totals = logs.groupby("PLAYER_ID")[SUM_COLS].sum()
    ids = ps.index.union(totals.index)
    gp = logs.groupby("PLAYER_ID").size().reindex(ids, fill_value=0)
    return blend(ps.reindex(ids).fillna(bench), per_game(totals.reindex(ids), gp), gp), bench


def expected_minutes(season):
    """Minutos esperados por partido de cada jugador, para sugerir la alineación.

    Promedia los minutos de las dos temporadas anteriores ponderando por partidos jugados, con 20
    partidos ficticios de 10 minutos para que una muestra corta no pese de más, y lo mezcla con los
    minutos de esta temporada, que pesan mucho desde el primer partido (k=1). Es la regla que mejor
    acertó a los 5 con más minutos de cada partido en 2024-25 y 2025-26 (3.97 de 5).
    """
    last = load("player_stats", prev_season(season)).set_index("PLAYER_ID")
    older = load("player_stats", prev_season(prev_season(season))).set_index("PLAYER_ID")
    total = (last["MIN"] * last["GP"]).add(older["MIN"] * older["GP"], fill_value=0)
    prev = (total + 10 * 20) / (last["GP"].add(older["GP"], fill_value=0) + 20)
    logs = regular_season_logs(season)
    ids = prev.index.union(logs["PLAYER_ID"].unique())
    gp = logs.groupby("PLAYER_ID").size().reindex(ids, fill_value=0)
    current = logs.groupby("PLAYER_ID")["MIN"].mean().reindex(ids)
    return blend(prev.reindex(ids).fillna(10), current, gp, k=1)


def lineup_stats(player_ids, ps, bench):
    """Promedio de stats de una alineación; los jugadores sin stats cuentan como banca."""
    return ps.reindex(player_ids).fillna(bench).mean()


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
        regular_season_logs(season)
        .sort_values("MIN", ascending=False)
        .groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .apply(lambda ids: list(ids[:5]))
    )


def build_dataset(season):
    """Una fila por partido con lo que se sabía antes de jugarlo."""
    logs = regular_season_logs(season)
    ps, bench = stats_table(prev_season(season))
    before = logs.groupby("PLAYER_ID")[SUM_COLS].cumsum() - logs[SUM_COLS]  # sin contar ese partido
    gp = logs.groupby("PLAYER_ID").cumcount()
    prev = ps.reindex(logs["PLAYER_ID"]).fillna(bench).set_axis(logs.index)
    stats = blend(prev, per_game(before, gp), gp)

    top5 = logs.sort_values("MIN", ascending=False).groupby(["GAME_ID", "TEAM_ID"]).head(5)
    lineups = stats.loc[top5.index].groupby([top5["GAME_ID"], top5["TEAM_ID"]]).mean()
    games = load("games", season)
    home = lineups.reindex(list(zip(games["GAME_ID"], games["HOME_TEAM_ID"]))).reset_index(drop=True)
    away = lineups.reindex(list(zip(games["GAME_ID"], games["AWAY_TEAM_ID"]))).reset_index(drop=True)
    return home - away, games["HOME_WIN"]


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
