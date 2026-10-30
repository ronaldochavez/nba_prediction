"""Entrena y evalúa la regresión logística que predice si gana el equipo local.

Cada partido se describe con la diferencia (local - visitante) del promedio de stats de los 5
jugadores con más minutos esperados de cada equipo entre los que jugaron (como los elegiría alguien
con el reporte de lesiones) y de los 3 siguientes (la banca), más el Elo de cada equipo y si jugó
el día anterior. Las stats de cada jugador mezclan su temporada anterior con lo que llevaba de la
actual antes de ese partido, igual que en la app.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import CURRENT_SEASON, SEASONS, STAT_COLS, load, prev_season

MODEL_PATH = Path(__file__).parent / "model.joblib"
BENCH_MIN = 15  # jugadores con menos minutos por partido = perfil de banca
TEST_SEASONS = 3  # temporadas finales que se usan para evaluar
# Partidos que "vale" la temporada anterior al mezclarla con la actual: a los 10 partidos jugados,
# mitad y mitad. Elegido probando 2, 5, 10, 20 y 40 en las temporadas 2023-24 a 2025-26.
K_GAMES = 10
SUM_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "PLUS_MINUS", "FGM", "FGA", "FG3M", "FG3A"]
BENCH_SIZE = 3  # jugadores de banca que cuentan: los 3 siguientes en minutos esperados
BENCH_COLS = ["BENCH_PTS", "BENCH_PLUS_MINUS"]
FEATURES = STAT_COLS + BENCH_COLS + ["ELO", "B2B_HOME", "B2B_AWAY"]
# Elo al estilo de FiveThirtyEight: 100 puntos de ventaja de local, 28 puntos de Elo ≈ 1 punto de
# margen, y al empezar cada temporada cada equipo regresa un 25 % hacia el promedio.
ELO_MEAN, ELO_K, ELO_HOME, ELO_PER_POINT = 1505, 20, 100, 28


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


def prior_minutes(season):
    """Minutos por partido esperados al empezar `season`, según las dos temporadas anteriores.

    Pondera por partidos jugados y suma 20 partidos ficticios de 10 minutos para que una muestra
    corta no pese de más.
    """
    last = load("player_stats", prev_season(season)).set_index("PLAYER_ID")
    older = load("player_stats", prev_season(prev_season(season))).set_index("PLAYER_ID")
    total = (last["MIN"] * last["GP"]).add(older["MIN"] * older["GP"], fill_value=0)
    return (total + 10 * 20) / (last["GP"].add(older["GP"], fill_value=0) + 20)


def expected_minutes(season):
    """Minutos esperados por partido de cada jugador, para sugerir la alineación.

    Mezcla los minutos de las dos temporadas anteriores con los de esta, que pesan mucho desde el
    primer partido (k=1). Es la regla que mejor acertó a los 5 con más minutos de cada partido en
    2024-25 y 2025-26 (3.97 de 5).
    """
    prev = prior_minutes(season)
    logs = regular_season_logs(season)
    ids = prev.index.union(logs["PLAYER_ID"].unique())
    gp = logs.groupby("PLAYER_ID").size().reindex(ids, fill_value=0)
    current = logs.groupby("PLAYER_ID")["MIN"].mean().reindex(ids)
    return blend(prev.reindex(ids).fillna(10), current, gp, k=1)


def lineup_stats(player_ids, ps, bench):
    """Promedio de stats de una alineación; los jugadores sin stats cuentan como banca."""
    return ps.reindex(player_ids).fillna(bench).mean()


def team_features(lineup_ids, bench_ids, ps, bench):
    """Stats promedio de la alineación y, con prefijo BENCH_, de los jugadores de banca."""
    reserves = ps.reindex(bench_ids).fillna(bench).mean() if len(bench_ids) else bench
    return pd.concat([lineup_stats(lineup_ids, ps, bench), reserves.add_prefix("BENCH_")[BENCH_COLS]])


def game_features(home, away, elo_diff, b2b_home, b2b_away):
    """Fila de features de un partido a partir de team_features de cada equipo."""
    return pd.concat([home - away, pd.Series({"ELO": elo_diff, "B2B_HOME": b2b_home, "B2B_AWAY": b2b_away})])[FEATURES]


def contributions(model, features):
    """Aporte de cada feature al puntaje del local: positivo favorece al local, negativo al visitante."""
    scaler, logreg = model[0], model[-1]
    return (features - scaler.mean_) / scaler.scale_ * logreg.coef_[0]


def weights(model):
    """Peso de cada feature (sobre features estandarizadas) y su importancia relativa."""
    coef = pd.Series(model[-1].coef_[0], index=model[0].feature_names_in_)
    importance = coef.abs() / coef.abs().sum()
    return pd.DataFrame({"Peso": coef, "Importancia": importance}).sort_values("Importancia", ascending=False)


def elo_ratings():
    """Elo de cada equipo partido a partido desde SEASONS[0] hasta hoy.

    Devuelve la diferencia local − visitante antes de cada partido (por GAME_ID) y el Elo actual de
    cada equipo. La primera temporada arranca del rating neto de la anterior. El cambio tras cada
    partido crece con el margen de victoria y es menor cuando gana el favorito.
    """
    first = load("team_ratings", prev_season(SEASONS[0])).set_index("TEAM_ID")
    elo = (ELO_MEAN + ELO_PER_POINT * (first["OFF_RATING"] - first["DEF_RATING"])).to_dict()
    before = {}
    for season in SEASONS + [CURRENT_SEASON]:
        elo = {team: 0.75 * e + 0.25 * ELO_MEAN for team, e in elo.items()}
        for g in load("games", season).sort_values(["GAME_DATE", "GAME_ID"]).itertuples():
            home, away = elo[g.HOME_TEAM_ID], elo[g.AWAY_TEAM_ID]
            before[g.GAME_ID] = home - away
            edge = home + ELO_HOME - away
            p_home = 1 / (1 + 10 ** (-edge / 400))
            margin = g.HOME_PTS - g.AWAY_PTS
            winner_edge = edge if margin > 0 else -edge
            shift = ELO_K * (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_edge) * ((margin > 0) - p_home)
            elo[g.HOME_TEAM_ID], elo[g.AWAY_TEAM_ID] = home + shift, away - shift
    return pd.Series(before), elo


def back_to_back(games):
    """Si el local y el visitante jugaron el día anterior (1) o no (0), con el mismo índice que `games`."""
    sides = pd.concat([
        pd.DataFrame({"ROW": games.index, "SIDE": side, "TEAM_ID": games[f"{side}_TEAM_ID"],
                      "DATE": pd.to_datetime(games["GAME_DATE"])})
        for side in ("HOME", "AWAY")
    ]).sort_values("DATE")
    sides["B2B"] = (sides.groupby("TEAM_ID")["DATE"].diff().dt.days == 1).astype(int)
    b2b = sides.pivot(index="ROW", columns="SIDE", values="B2B").reindex(games.index)
    return b2b["HOME"], b2b["AWAY"]


def game_players(season):
    """Box scores de temporada regular con las stats y los minutos esperados de cada jugador antes del partido.

    RANK ordena a los que jugaron cada partido por minutos esperados: 1 a 5 son la alineación y los
    BENCH_SIZE siguientes, la banca.
    """
    logs = regular_season_logs(season)
    ps, bench = stats_table(prev_season(season))
    before = logs.groupby("PLAYER_ID")[SUM_COLS].cumsum() - logs[SUM_COLS]  # sin contar ese partido
    gp = logs.groupby("PLAYER_ID").cumcount()
    prev = ps.reindex(logs["PLAYER_ID"]).fillna(bench).set_axis(logs.index)
    stats = blend(prev, per_game(before, gp), gp)
    prev_min = prior_minutes(season).reindex(logs["PLAYER_ID"]).fillna(10).set_axis(logs.index)
    expected = blend(prev_min, before["MIN"] / gp.replace(0, np.nan), gp, k=1)
    rank = expected.groupby([logs["GAME_ID"], logs["TEAM_ID"]]).rank(ascending=False, method="first")
    return logs[["GAME_ID", "TEAM_ID", "PLAYER_ID"]].assign(RANK=rank), stats, bench


def game_lineups(season):
    """Los 5 jugadores con más minutos esperados de cada equipo en cada partido, por (GAME_ID, TEAM_ID)."""
    players, _, _ = game_players(season)
    top5 = players[players["RANK"] <= 5]
    return top5.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"].apply(list)


def build_dataset(season):
    """Una fila por partido con lo que se sabía antes de jugarlo."""
    players, stats, bench = game_players(season)
    key = [players["GAME_ID"], players["TEAM_ID"]]
    lineup = players["RANK"] <= 5
    reserves = (players["RANK"] > 5) & (players["RANK"] <= 5 + BENCH_SIZE)
    teams = stats[lineup].groupby([k[lineup] for k in key]).mean().join(
        stats[reserves].groupby([k[reserves] for k in key]).mean().add_prefix("BENCH_")[BENCH_COLS]
    )
    teams[BENCH_COLS] = teams[BENCH_COLS].fillna(bench.add_prefix("BENCH_")[BENCH_COLS])  # equipos con pocos jugadores

    games = load("games", season)
    home = teams.reindex(list(zip(games["GAME_ID"], games["HOME_TEAM_ID"]))).reset_index(drop=True)
    away = teams.reindex(list(zip(games["GAME_ID"], games["AWAY_TEAM_ID"]))).reset_index(drop=True)
    X = home - away
    X["ELO"] = games["GAME_ID"].map(elo_ratings()[0])
    X["B2B_HOME"], X["B2B_AWAY"] = back_to_back(games)
    return X[FEATURES], games["HOME_WIN"]


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
    print(f"{'Prueba':<9}{'Entrenamiento':<22}{'Modelo':>7}{'Log loss':>10}{'Siempre local':>15}")
    accuracies, losses, baselines = [], [], []
    for i in range(len(SEASONS) - TEST_SEASONS, len(SEASONS)):
        model = train([datasets[s] for s in SEASONS[:i]])
        X, y = datasets[SEASONS[i]]
        accuracies.append(accuracy_score(y, model.predict(X)))
        losses.append(log_loss(y, model.predict_proba(X)[:, 1]))  # calidad de las probabilidades: menos es mejor
        baselines.append(y.mean())  # "siempre gana el local"
        print(
            f"{SEASONS[i]:<9}{SEASONS[0] + ' a ' + SEASONS[i - 1]:<22}{accuracies[-1]:>7.1%}{losses[-1]:>10.4f}"
            f"{baselines[-1]:>15.1%}"
        )
    print(f"{'Promedio':<31}{np.mean(accuracies):>7.1%}{np.mean(losses):>10.4f}{np.mean(baselines):>15.1%}")

    model = train(datasets.values())
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo final: {sum(len(y) for _, y in datasets.values())} partidos de {SEASONS[0]} a {SEASONS[-1]}")
    print(weights(model).to_string(formatters={"Peso": "{:+.2f}".format, "Importancia": "{:.0%}".format}))
    print(f"Ventaja de local: {model[-1].intercept_[0]:+.2f}")
    print(f"Guardado en {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
