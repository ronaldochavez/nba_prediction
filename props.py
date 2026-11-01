"""Proyecta los puntos de cada jugador en un partido para compararlos con las líneas de las casas (props).

    proyección = 70 % × puntos por minuto × minutos esperados × defensa del rival × ritmo del partido
               + 30 % × promedio de sus últimos 10 partidos de la temporada
               + β × puntos que dejan los compañeros habituales ausentes × su parte de los puntos del equipo

La defensa del rival es su rating defensivo entre el de la liga, y el ritmo, las posesiones que se esperan
en el partido (como en points.py) entre las de su equipo. Un compañero habitual promedió al menos 20
minutos en los últimos 10 partidos del equipo. La probabilidad de over sale de los errores reales en
partidos pasados, porque los puntos no se reparten simétricos: hay más noches flojas y unas pocas muy
altas. `python props.py` evalúa la proyección y guarda β y esa distribución en props_model.json.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import CURRENT_SEASON, DATA_DIR, SEASONS, load
from model import game_players, regular_season_logs
from points import DEVIATION_WEIGHT, league_to_date, team_ratings_to_date

PROPS_MODEL_PATH = Path(__file__).parent / "props_model.json"
# Líneas que escribes en la pestaña Props; es un archivo personal, no se sube a GitHub
PROPS_PATH = DATA_DIR / f"props_{CURRENT_SEASON}.csv"
PROPS_COLS = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM", "LINE", "ODDS_OVER", "ODDS_UNDER", "PROJ", "P_OVER",
    "SAVED_AT",
]
FORM_WEIGHT = 0.3  # peso de los últimos 10 partidos; probado con 10 a 40 % en 2023-24 a 2025-26
REGULAR_MIN = 20  # minutos por partido de un compañero habitual y de los jugadores que se evalúan
RECENT_GAMES = 10
QUANTILES = np.linspace(0, 1, 201)


def project(df, beta):
    """Puntos proyectados a partir de las columnas PPM, EXP_MIN, OPP_DEF, PACE_F, LAST10, MISSING y SHARE."""
    matchup = df["PPM"] * df["EXP_MIN"] * df["OPP_DEF"] * df["PACE_F"]
    return (1 - FORM_WEIGHT) * matchup + FORM_WEIGHT * df["LAST10"].fillna(matchup) + beta * df["MISSING"] * df["SHARE"]


def missing_points(team_logs):
    """Puntos por partido de los compañeros habituales que no juegan, por GAME_ID de un equipo.

    Habitual = promedió al menos REGULAR_MIN minutos en los RECENT_GAMES partidos anteriores del equipo,
    contando con cero los partidos que no jugó.
    """
    game_ids = team_logs.drop_duplicates("GAME_ID").sort_values(["GAME_DATE", "GAME_ID"])["GAME_ID"].tolist()
    result = {}
    for i, game_id in enumerate(game_ids):
        recent = team_logs[team_logs["GAME_ID"].isin(game_ids[max(0, i - RECENT_GAMES):i])]
        n = recent["GAME_ID"].nunique()
        if not n:
            result[game_id] = 0.0
            continue
        per_game = recent.groupby("PLAYER_ID")[["MIN", "PTS"]].sum() / n
        played = set(team_logs.loc[team_logs["GAME_ID"] == game_id, "PLAYER_ID"])
        absent = per_game[(per_game["MIN"] >= REGULAR_MIN) & ~per_game.index.isin(played)]
        result[game_id] = absent["PTS"].sum()
    return result


def season_rows(season):
    """Una fila por jugador de rotación y partido con lo que se sabía antes de jugarlo y los puntos que hizo."""
    logs = regular_season_logs(season)
    players, stats, _ = game_players(season)
    df = logs[["GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID", "PTS"]].assign(
        BASE=stats["PTS"], PPM=stats["PTS"] / stats["MIN"], EXP_MIN=players["EXPECTED_MIN"],
        LAST10=logs.groupby("PLAYER_ID")["PTS"].transform(lambda s: s.shift().rolling(RECENT_GAMES, min_periods=3).mean()),
    )
    games = load("games", season)
    opponents = pd.concat([
        games[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]].set_axis(["GAME_ID", "TEAM_ID", "OPP_ID"], axis=1),
        games[["GAME_ID", "AWAY_TEAM_ID", "HOME_TEAM_ID"]].set_axis(["GAME_ID", "TEAM_ID", "OPP_ID"], axis=1),
    ])
    df = df.merge(opponents, on=["GAME_ID", "TEAM_ID"])
    to_date = team_ratings_to_date(season)
    league = league_to_date(to_date, games).set_axis(games["GAME_ID"]).reindex(df["GAME_ID"]).reset_index(drop=True)
    own = to_date.reindex(list(zip(df["GAME_ID"], df["TEAM_ID"]))).reset_index(drop=True)
    opp = to_date.reindex(list(zip(df["GAME_ID"], df["OPP_ID"]))).reset_index(drop=True)
    poss = league["PACE"] + DEVIATION_WEIGHT * (own["PACE"] + opp["PACE"] - 2 * league["PACE"])
    df["OPP_DEF"] = (opp["DEF_RATING"] / league["DEF_RATING"]).values
    df["PACE_F"] = (poss / own["PACE"]).values
    missing = {}
    for team_id, team_logs in logs.groupby("TEAM_ID"):
        missing.update({(g, team_id): pts for g, pts in missing_points(team_logs).items()})
    df["MISSING"] = [missing.get(key, 0.0) for key in zip(df["GAME_ID"], df["TEAM_ID"])]
    df = df[df["EXP_MIN"] >= REGULAR_MIN].dropna(subset=["OPP_DEF", "PACE_F", "PPM"])
    df["SHARE"] = df["BASE"] / df.groupby(["GAME_ID", "TEAM_ID"])["BASE"].transform("sum")
    return df.assign(SEASON=season)


def fit(rows):
    """β de los ausentes, ancho de los errores según la proyección y su distribución (estandarizada)."""
    base = project(rows.assign(MISSING=0.0), 0.0)
    boost = rows["MISSING"] * rows["SHARE"]
    beta = ((rows["PTS"] - base) * boost).sum() / (boost**2).sum()
    proj = project(rows, beta)
    resid = rows["PTS"] - proj
    bins = pd.qcut(proj, 10)
    sd_b, sd_a = np.polyfit(proj.groupby(bins, observed=True).mean(), resid.groupby(bins, observed=True).std(), 1)
    z = resid / (sd_a + sd_b * proj)
    return {"beta": beta, "sd_a": sd_a, "sd_b": sd_b, "z": np.quantile(z, QUANTILES).tolist()}


def p_over(proj, line, params):
    """Probabilidad de que el jugador anote más que `line`."""
    z = (line - proj) / (params["sd_a"] + params["sd_b"] * proj)
    return 1 - np.interp(z, params["z"], QUANTILES)


def load_params():
    return json.loads(PROPS_MODEL_PATH.read_text()) if PROPS_MODEL_PATH.exists() else None


def player_projections(team_id, opp_id, available, absent, ps, bench, minutes, team_r, logs, params):
    """Proyección de puntos de los jugadores disponibles de un equipo en su próximo partido.

    `absent` son los que no juegan (lesionados o inactivos), `ps` y `minutes` las stats y minutos esperados
    de model.py, `team_r` los ratings de points.py y `logs` los box scores de temporada regular actuales.
    """
    league = team_r.mean()
    poss = league["PACE"] + DEVIATION_WEIGHT * (team_r.loc[team_id, "PACE"] + team_r.loc[opp_id, "PACE"] - 2 * league["PACE"])
    stats = ps.reindex(available).fillna(bench)
    team_logs = logs[logs["TEAM_ID"] == team_id]
    recent_ids = team_logs.drop_duplicates("GAME_ID").sort_values(["GAME_DATE", "GAME_ID"])["GAME_ID"].tail(RECENT_GAMES)
    recent = team_logs[team_logs["GAME_ID"].isin(recent_ids)]
    per_game = recent.groupby("PLAYER_ID")[["MIN", "PTS"]].sum() / max(len(recent_ids), 1)
    regulars = per_game[per_game["MIN"] >= REGULAR_MIN]
    player_logs = logs[logs["PLAYER_ID"].isin(available)].sort_values(["GAME_DATE", "GAME_ID"]).groupby("PLAYER_ID")["PTS"]
    last10 = player_logs.apply(lambda s: s.tail(RECENT_GAMES).mean() if len(s) >= 3 else np.nan)
    df = pd.DataFrame({
        "PLAYER_ID": available,
        "BASE": stats["PTS"].values,
        "PPM": (stats["PTS"] / stats["MIN"]).values,
        "EXP_MIN": minutes.reindex(available).fillna(10).values,
        "OPP_DEF": team_r.loc[opp_id, "DEF_RATING"] / league["DEF_RATING"],
        "PACE_F": poss / team_r.loc[team_id, "PACE"],
        "LAST10": pd.Series(available).map(last10),
        "MISSING": regulars.loc[regulars.index.isin(absent), "PTS"].sum(),
    })
    rotation = df["EXP_MIN"] >= REGULAR_MIN
    df["SHARE"] = df["BASE"] / (df.loc[rotation, "BASE"].sum() or df["BASE"].sum())
    df["PROJ"] = project(df, params["beta"])
    return df.sort_values("PROJ", ascending=False).reset_index(drop=True)


def load_props():
    if not PROPS_PATH.exists():
        return pd.DataFrame(columns=PROPS_COLS)
    return pd.read_csv(PROPS_PATH, dtype={"GAME_ID": str})


def save_prop(game_id, game_date, player_id, name, team, line, odds_over, odds_under, proj, prob_over):
    """Guarda la línea de un jugador con la proyección del momento; reemplaza la anterior. Sin línea, la borra."""
    log = load_props()
    log = log[~((log["GAME_ID"] == game_id) & (log["PLAYER_ID"] == player_id))]
    if line is not None:
        row = pd.DataFrame([[
            game_id, game_date, player_id, name, team, line, odds_over, odds_under, round(proj, 2), round(prob_over, 4),
            dt.datetime.now().isoformat(timespec="seconds"),
        ]], columns=PROPS_COLS)
        log = pd.concat([log, row] if len(log) else [row], ignore_index=True)
    log.to_csv(PROPS_PATH, index=False)


def props_results(props, logs):
    """Props guardados de partidos ya jugados con el pick (el lado más probable), si se cumplió y lo que dejó
    apostar 1 unidad. Si el jugador no jugó, la apuesta se anula y no aparece; si anota justo la línea, se
    devuelve (HIT vacío y 0 de ganancia)."""
    df = props.merge(logs[["GAME_ID", "PLAYER_ID", "PTS"]], on=["GAME_ID", "PLAYER_ID"])
    df["PICK"] = np.where(df["P_OVER"] >= 0.5, "Over", "Under")
    over = df["PICK"] == "Over"
    df["HIT"] = np.where(over, df["PTS"] > df["LINE"], df["PTS"] < df["LINE"]).astype(float)
    push = df["PTS"] == df["LINE"]
    df.loc[push, "HIT"] = np.nan
    odds = np.where(over, df["ODDS_OVER"], df["ODDS_UNDER"])
    df["PROFIT"] = np.where(push, 0.0, np.where(df["HIT"] == 1, odds - 1, -1.0))
    return df


def main():
    rows = pd.concat([season_rows(s) for s in SEASONS[1:]], ignore_index=True)  # 2021-22 no tiene ratings previos
    train, test = rows[rows["SEASON"] == SEASONS[1]], rows[rows["SEASON"].isin(SEASONS[-3:])]
    params = fit(train)
    proj = project(test, params["beta"])
    print(f"Puntos de jugadores con {REGULAR_MIN}+ minutos esperados, {SEASONS[-3]} a {SEASONS[-1]} "
          f"({len(test)} jugador-partidos), con parámetros de {SEASONS[1]}\n")
    print(f"  Promedio de la temporada: error {(test['PTS'] - test['BASE']).abs().mean():.2f} puntos")
    print(f"  Proyección:               error {(test['PTS'] - proj).abs().mean():.2f} puntos, "
          f"sesgo {(proj - test['PTS']).mean():+.2f}")
    # Calibración con una línea típica: el promedio de la temporada redondeado a x.5
    line = np.floor(test["BASE"]) + 0.5
    calib = pd.DataFrame({"p": p_over(proj, line, params), "over": test["PTS"] > line})
    table = calib.groupby(pd.cut(calib["p"], [0, .35, .45, .55, .65, 1]), observed=True).agg(
        partidos=("over", "size"), predicho=("p", "mean"), real=("over", "mean")
    )
    print("\n  Probabilidad de over con la línea en el promedio de la temporada (x.5):")
    print(table.round(3).to_string())

    params = fit(rows)
    PROPS_MODEL_PATH.write_text(json.dumps(params))
    print(f"\nParámetros finales ({SEASONS[1]} a {SEASONS[-1]}): β {params['beta']:.2f}, ancho de los errores "
          f"{params['sd_a']:.2f} + {params['sd_b']:.3f} × proyección. Guardado en {PROPS_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
