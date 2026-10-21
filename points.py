"""Proyecta los puntos de cada equipo y el total del partido con efectividad × posesiones.

Puntos de A contra B = posesiones × (rating ofensivo de A + rating defensivo de B) / 2 / 100,
con posesiones = promedio del ritmo (PACE) de ambos. Al local se le suma la mitad del margen
medio de los locales y al visitante se le resta. Se calcula de dos formas, siempre con los
ratings de la temporada anterior:
- por equipo: ratings del equipo completo;
- por alineación: promedio de los ratings en cancha de los 5 jugadores.
"""
import pandas as pd

from data import RATING_COLS, SEASONS, load, prev_season
from model import BENCH_MIN, TEST_SEASONS, game_lineups

SHRINK_POSS = 500  # posesiones de promedio de liga que se suman a las de cada jugador


def ratings_tables(season):
    """Ratings de equipos, de jugadores y de banca de una temporada, y el margen medio del local."""
    teams = load("team_ratings", season).set_index("TEAM_ID")[RATING_COLS]
    league = teams.mean()
    players = load("player_ratings", season).set_index("PLAYER_ID")
    # Con pocas posesiones el rating de un jugador es casi puro ruido: se mezcla con el de la liga
    players = (players[RATING_COLS].mul(players["POSS"], axis=0) + league * SHRINK_POSS).div(
        players["POSS"] + SHRINK_POSS, axis=0
    )
    stats = load("player_stats", season).set_index("PLAYER_ID")
    bench = players.reindex(stats.index[stats["MIN"] < BENCH_MIN]).mean()
    games = load("games", season)
    home_edge = (games["HOME_PTS"] - games["AWAY_PTS"]).mean()
    return teams, players, bench, home_edge


def lineup_ratings(player_ids, players, bench):
    """Promedio de los ratings de una alineación; los jugadores sin stats cuentan como banca."""
    return players.reindex(player_ids).fillna(bench).mean()


def project(home, away, home_edge):
    """Puntos proyectados de local y visitante a partir de sus ratings."""
    poss = (home["PACE"] + away["PACE"]) / 2
    home_pts = poss * (home["OFF_RATING"] + away["DEF_RATING"]) / 200 + home_edge / 2
    away_pts = poss * (away["OFF_RATING"] + home["DEF_RATING"]) / 200 - home_edge / 2
    return home_pts, away_pts


def season_projections(season):
    """Proyecciones de cada partido de una temporada: promedio de la liga, por equipo y por alineación."""
    games = load("games", season)
    teams, players, bench, home_edge = ratings_tables(prev_season(season))
    lineups = game_lineups(season)

    def lineup_table(side):
        ids = [lineups[gid, tid] for gid, tid in zip(games["GAME_ID"], games[f"{side}_TEAM_ID"])]
        return pd.DataFrame([lineup_ratings(i, players, bench) for i in ids])

    def team_table(side):
        return teams.reindex(games[f"{side}_TEAM_ID"]).reset_index(drop=True)

    league = teams.mean()
    return games, {
        "Promedio de la liga": project(league, league, home_edge),
        "Por equipo": project(team_table("HOME"), team_table("AWAY"), home_edge),
        "Por alineación": project(lineup_table("HOME"), lineup_table("AWAY"), home_edge),
    }


def main():
    team_errors, total_errors = {}, {}
    for season in SEASONS[-TEST_SEASONS:]:
        games, projections = season_projections(season)
        for name, (home, away) in projections.items():
            team_errors.setdefault(name, []).extend([home - games["HOME_PTS"], away - games["AWAY_PTS"]])
            total_errors.setdefault(name, []).append(home + away - games["HOME_PTS"] - games["AWAY_PTS"])

    results = pd.DataFrame({
        name: {
            "Error por equipo": pd.concat(team_errors[name]).abs().mean(),
            "Error del total": pd.concat(total_errors[name]).abs().mean(),
            "Sesgo del total": pd.concat(total_errors[name]).mean(),
        }
        for name in team_errors
    }).T
    print(f"Proyección de puntos de {SEASONS[-TEST_SEASONS]} a {SEASONS[-1]}, cada temporada con los ratings de la anterior")
    print("(error = diferencia media en puntos con el marcador real; sesgo negativo = se proyecta de menos)\n")
    print(results.round(2).to_string())


if __name__ == "__main__":
    main()
