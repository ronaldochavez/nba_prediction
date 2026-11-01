"""Proyecta los puntos de cada equipo y el total del partido con efectividad × posesiones.

Cada equipo parte del promedio de la liga y se le suma lo que su ataque está por encima del promedio
y lo que la defensa rival está por debajo (lo mismo con el ritmo para las posesiones):
    posesiones = liga + 0.9 × (ritmo A − liga) + 0.9 × (ritmo B − liga)
    puntos de A = posesiones × (liga + 0.9 × (ataque A − liga) + 0.9 × (defensa B − liga)) / 100
Al local se le suma la mitad del margen medio de los locales y al visitante se le resta; el equipo que
jugó el día anterior anota menos y su rival más.
Se calcula de dos formas:
- por equipo: ratings del equipo, mezclando la temporada anterior con lo que va de la actual
  (calculado con los box scores, partido a partido);
- por alineación: promedio de los ratings en cancha de los 5 jugadores. Los de la temporada
  actual solo existen acumulados a hoy, así que la mezcla se aplica en la app pero la
  evaluación histórica usa solo la temporada anterior.
"""
import pandas as pd

from data import RATING_COLS, SEASONS, load, prev_season
from model import BENCH_MIN, TEST_SEASONS, back_to_back, blend, game_lineups, regular_season_logs

SHRINK_POSS = 500  # posesiones de promedio de liga que se suman a las de cada jugador
TEAM_SUMS = ["PTS", "POSS", "OPP_PTS", "OPP_POSS", "MIN"]
# Parte de la diferencia con la liga que cuenta: los ratings tienen ruido y con 0.9 la diferencia y el
# total proyectados quedan en la misma escala que los reales (con 1 se pasan un 10 %). Probado en 2023-24 a 2025-26.
DEVIATION_WEIGHT = 0.9
# Puntos de más o de menos cuando un equipo jugó el día anterior, medidos en 2022-23 a 2025-26
B2B_PTS, B2B_RIVAL_PTS = -1.5, 1.2


def player_ratings(season):
    """Ratings en cancha de cada jugador, mezclados con el promedio de la liga según sus posesiones.

    Con pocas posesiones el rating de un jugador es casi puro ruido.
    """
    league = load("team_ratings", season)[RATING_COLS].mean()
    players = load("player_ratings", season).set_index("PLAYER_ID")
    return (players[RATING_COLS].mul(players["POSS"], axis=0) + league * SHRINK_POSS).div(
        players["POSS"] + SHRINK_POSS, axis=0
    )


def ratings_tables(season):
    """Ratings de equipos, de jugadores y de banca de una temporada, y el margen medio del local."""
    teams = load("team_ratings", season).set_index("TEAM_ID")[RATING_COLS]
    players = player_ratings(season)
    stats = load("player_stats", season).set_index("PLAYER_ID")
    bench = players.reindex(stats.index[stats["MIN"] < BENCH_MIN]).mean()
    # Si un jugador no jugó esa temporada (p. ej. por una lesión), se usan sus ratings de la anterior
    players = players.combine_first(player_ratings(prev_season(season)))
    games = load("games", season)
    home_edge = (games["HOME_PTS"] - games["AWAY_PTS"]).mean()
    return teams, players, bench, home_edge


def team_games(season):
    """Una fila por equipo y partido, en orden cronológico, con puntos y posesiones propias y del rival."""
    logs = regular_season_logs(season)
    t = logs.groupby(["GAME_ID", "TEAM_ID"], sort=False)[["PTS", "FGA", "FTA", "OREB", "TOV", "MIN"]].sum().reset_index()
    t["POSS"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp = t[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS", "POSS": "OPP_POSS"})
    return t.merge(opp, on="GAME_ID").query("TEAM_ID != OPP_ID").reset_index(drop=True)


def box_ratings(totals):
    """Ratings ofensivo, defensivo y ritmo a partir de totales de box score."""
    return pd.DataFrame({
        "OFF_RATING": 100 * totals["PTS"] / totals["POSS"],
        "DEF_RATING": 100 * totals["OPP_PTS"] / totals["OPP_POSS"],
        "PACE": 48 * (totals["POSS"] + totals["OPP_POSS"]) / 2 / (totals["MIN"] / 5),
    })


def team_ratings_to_date(season):
    """Ratings de cada equipo antes de cada partido, por (GAME_ID, TEAM_ID)."""
    t = team_games(season)
    before = t.groupby("TEAM_ID")[TEAM_SUMS].cumsum() - t[TEAM_SUMS]  # sin contar ese partido
    gp = t.groupby("TEAM_ID").cumcount()
    prev = ratings_tables(prev_season(season))[0].reindex(t["TEAM_ID"]).set_axis(t.index)
    return blend(prev, box_ratings(before), gp).set_axis(pd.MultiIndex.from_frame(t[["GAME_ID", "TEAM_ID"]]))


def current_ratings(season):
    """Ratings de equipos y jugadores para predecir los próximos partidos de `season`."""
    teams, players, bench, home_edge = ratings_tables(prev_season(season))
    t = team_games(season)
    gp = t.groupby("TEAM_ID").size().reindex(teams.index, fill_value=0)
    teams = blend(teams, box_ratings(t.groupby("TEAM_ID")[TEAM_SUMS].sum().reindex(teams.index)), gp)
    current = load("player_ratings", season).set_index("PLAYER_ID")
    ids = players.index.union(current.index)
    gp = current["GP"].reindex(ids, fill_value=0)
    players = blend(players.reindex(ids).fillna(bench), current[RATING_COLS].reindex(ids), gp)
    return teams, players, bench, home_edge


def lineup_ratings(player_ids, players, bench):
    """Promedio de los ratings de una alineación; los jugadores sin stats cuentan como banca."""
    return players.reindex(player_ids).fillna(bench).mean()


def project(home, away, league, home_edge, b2b_home=0, b2b_away=0):
    """Puntos proyectados de local y visitante a partir de sus ratings y de los de la liga.

    `b2b_home` y `b2b_away` valen 1 si ese equipo jugó el día anterior.
    """
    def points(attack, defense):
        poss = league["PACE"] + DEVIATION_WEIGHT * (attack["PACE"] + defense["PACE"] - 2 * league["PACE"])
        efficiency = league["OFF_RATING"] + DEVIATION_WEIGHT * (
            attack["OFF_RATING"] - league["OFF_RATING"] + defense["DEF_RATING"] - league["DEF_RATING"]
        )
        return poss * efficiency / 100

    return (
        points(home, away) + home_edge / 2 + B2B_PTS * b2b_home + B2B_RIVAL_PTS * b2b_away,
        points(away, home) - home_edge / 2 + B2B_PTS * b2b_away + B2B_RIVAL_PTS * b2b_home,
    )


def league_to_date(to_date, games):
    """Ratings de la liga antes de cada partido: el último rating previo de cada equipo, promediado entre los 30."""
    rows = to_date.reset_index().merge(games[["GAME_ID", "GAME_DATE"]], on="GAME_ID")
    by_date = pd.DataFrame({
        col: rows.pivot_table(index="GAME_DATE", columns="TEAM_ID", values=col).ffill().mean(axis=1) for col in RATING_COLS
    })
    return by_date.reindex(games["GAME_DATE"]).reset_index(drop=True)


def season_projections(season):
    """Proyecciones de cada partido de una temporada: promedio de la liga, por equipo y por alineación."""
    games = load("games", season)
    teams, players, bench, home_edge = ratings_tables(prev_season(season))
    to_date = team_ratings_to_date(season)
    lineups = game_lineups(season)

    def team_table(side):
        return to_date.reindex(list(zip(games["GAME_ID"], games[f"{side}_TEAM_ID"]))).reset_index(drop=True)

    def lineup_table(side):
        keys = zip(games["GAME_ID"], games[f"{side}_TEAM_ID"])
        return pd.DataFrame([lineup_ratings(lineups.get(key, []), players, bench) for key in keys])

    league = teams.mean()  # la evaluación por alineación usa solo la temporada anterior
    b2b_home, b2b_away = back_to_back(games)
    return games, {
        "Promedio de la liga": project(league, league, league, home_edge),
        "Por equipo": project(
            team_table("HOME"), team_table("AWAY"), league_to_date(to_date, games), home_edge, b2b_home, b2b_away
        ),
        "Por alineación": project(lineup_table("HOME"), lineup_table("AWAY"), league, home_edge, b2b_home, b2b_away),
    }


def main():
    projected, real = {}, []
    for season in SEASONS[-TEST_SEASONS:]:
        games, projections = season_projections(season)
        real.append(games)
        for name, (home, away) in projections.items():
            projected.setdefault(name, []).append(pd.DataFrame({"HOME": home, "AWAY": away}, index=games.index))
    games = pd.concat(real, ignore_index=True)
    real_margin = games["HOME_PTS"] - games["AWAY_PTS"]

    results = {}
    for name, parts in projected.items():
        p = pd.concat(parts, ignore_index=True)
        team_err = pd.concat([p["HOME"] - games["HOME_PTS"], p["AWAY"] - games["AWAY_PTS"]])
        total_err = p["HOME"] + p["AWAY"] - games["HOME_PTS"] - games["AWAY_PTS"]
        margin = p["HOME"] - p["AWAY"]
        results[name] = {
            "Error por equipo": team_err.abs().mean(),
            "Error del total": total_err.abs().mean(),
            "Sesgo del total": total_err.mean(),
            "Error de la diferencia": (margin - real_margin).abs().mean(),
            # Cuánto hay que multiplicar la diferencia proyectada para llegar a la real en promedio (1 = escala correcta)
            "Escala de la diferencia": margin.cov(real_margin) / margin.var() if margin.std() > 1 else None,
        }
    print(f"Proyección de puntos de {SEASONS[-TEST_SEASONS]} a {SEASONS[-1]}, con lo que se sabía antes de cada partido")
    print("(error = diferencia media en puntos con el marcador real; sesgo negativo = se proyecta de menos)\n")
    print(pd.DataFrame(results).T.round(2).to_string())


if __name__ == "__main__":
    main()
