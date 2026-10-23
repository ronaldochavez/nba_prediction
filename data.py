"""Descarga los datos de la NBA con nba_api y los guarda en CSV dentro de data/."""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelogs,
    scheduleleaguev2,
)
from nba_api.stats.static import teams as nba_teams

DATA_DIR = Path(__file__).parent / "data"
MOVES_PATH = Path(__file__).parent / "roster_moves.csv"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]  # temporadas con partidos para entrenar
CURRENT_SEASON = "2026-27"
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]
# Puntos anotados / recibidos por 100 posesiones y posesiones por 48 minutos
RATING_COLS = ["OFF_RATING", "DEF_RATING", "PACE"]
PAUSE = 1.5  # segundos entre llamadas para no saturar stats.nba.com


def fetch(endpoint, retries=3, **params):
    """Llama a un endpoint con pausa previa y reintentos; devuelve su primer DataFrame."""
    for attempt in range(1, retries + 1):
        time.sleep(PAUSE)
        try:
            return endpoint(timeout=60, **params).get_data_frames()[0]
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  error ({e}), reintento {attempt}...")
            time.sleep(PAUSE * 5)


def player_stats(season):
    df = fetch(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    )
    return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP"] + STAT_COLS]


def player_ratings(season):
    """Ratings del equipo mientras cada jugador está en cancha."""
    df = fetch(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    return df[["PLAYER_ID", "PLAYER_NAME", "GP", "POSS"] + RATING_COLS]


def team_ratings(season):
    df = fetch(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    return df[["TEAM_ID", "TEAM_NAME", "GP"] + RATING_COLS]


def game_logs(season):
    """Box score de cada jugador en cada partido: pretemporada, temporada regular y playoffs."""
    frames = [
        fetch(playergamelogs.PlayerGameLogs, season_nullable=season, season_type_nullable=season_type)
        .assign(SEASON_TYPE=season_type)
        for season_type in ["Pre Season", "Regular Season", "Playoffs"]
    ]
    df = pd.concat(frames, ignore_index=True)
    # En pretemporada también juegan equipos internacionales: solo se dejan los de la NBA
    df = df[df["TEAM_ID"].isin({t["id"] for t in nba_teams.get_teams()})]
    return df[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
        "SEASON_TYPE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTA", "OREB",
        "PLUS_MINUS",
    ]]


def season_games(season):
    """Partidos de temporada regular según el calendario oficial.

    Se usa el calendario y no LeagueGameLog porque en partidos de sede neutral
    (México, París, Copa NBA) el game log marca a ambos equipos como visitantes.
    """
    df = fetch(scheduleleaguev2.ScheduleLeagueV2, season=season)
    df = df[df["gameId"].str.startswith("002")]  # 002 = temporada regular
    return pd.DataFrame({
        "GAME_ID": df["gameId"],
        "GAME_DATE": df["gameDateEst"].str[:10],
        "HOME_TEAM_ID": df["homeTeam_teamId"],
        "HOME_TEAM": df["homeTeam_teamTricode"],
        "AWAY_TEAM_ID": df["awayTeam_teamId"],
        "AWAY_TEAM": df["awayTeam_teamTricode"],
        "HOME_PTS": df["homeTeam_score"],
        "AWAY_PTS": df["awayTeam_score"],
        "STATUS": df["gameStatus"],  # 3 = terminado
    })


def games(season):
    """Partidos ya jugados con su resultado."""
    df = season_games(season)
    df = df[df["STATUS"] == 3].drop(columns="STATUS")
    df["HOME_WIN"] = (df["HOME_PTS"] > df["AWAY_PTS"]).astype(int)
    return df


def schedule(season):
    return season_games(season).drop(columns=["HOME_PTS", "AWAY_PTS", "STATUS"])


def rosters(season):
    df = fetch(commonallplayers.CommonAllPlayers, is_only_current_season=1, season=season)
    df = df[df["TEAM_ID"] != 0].rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})
    return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]


def save(name, download, season, refresh=False):
    """Descarga y guarda un CSV; si ya existe solo lo vuelve a bajar con refresh=True."""
    path = DATA_DIR / f"{name}_{season}.csv"
    if path.exists() and not refresh:
        print(f"{path.name}: ya existe, se omite")
        return
    df = download(season)
    df.to_csv(path, index=False)
    print(f"{path.name}: {len(df)} filas")


def load(name, season):
    # GAME_ID como texto para conservar los ceros iniciales (0022400001)
    return pd.read_csv(DATA_DIR / f"{name}_{season}.csv", dtype={"GAME_ID": str})


def current_rosters():
    """Plantillas de la API más los movimientos de roster_moves.csv que la API aún no refleja.

    Un movimiento solo se aplica si la API todavía tiene al jugador en FROM_TEAM
    (vacío = no está en ninguna plantilla), así que deja de aplicarse cuando la API se actualiza.
    TO_TEAM vacío = el jugador sale de la plantilla.
    """
    rosters = load("rosters", CURRENT_SEASON)
    moves = pd.read_csv(MOVES_PATH).fillna({"FROM_TEAM": "", "TO_TEAM": ""})
    team_of = dict(zip(rosters["PLAYER_ID"], rosters["TEAM_ABBREVIATION"]))
    team_ids = dict(zip(rosters["TEAM_ABBREVIATION"], rosters["TEAM_ID"]))

    pending = moves[moves["PLAYER_ID"].map(team_of).fillna("") == moves["FROM_TEAM"]]
    arrivals = pending[pending["TO_TEAM"] != ""]
    rosters = rosters[~rosters["PLAYER_ID"].isin(pending["PLAYER_ID"])]
    return pd.concat([rosters, pd.DataFrame({
        "PLAYER_ID": arrivals["PLAYER_ID"],
        "PLAYER_NAME": arrivals["PLAYER_NAME"],
        "TEAM_ID": arrivals["TO_TEAM"].map(team_ids),
        "TEAM_ABBREVIATION": arrivals["TO_TEAM"],
    })], ignore_index=True)


def prev_season(season):
    """'2024-25' -> '2023-24'"""
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[2:]}"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    # Cada partido se predice con stats de la temporada anterior (o la previa, si el jugador
    # no jugó esa), así que hacen falta las dos temporadas previas a la primera
    first = prev_season(SEASONS[0])
    for season in [prev_season(first), first] + SEASONS:
        save("player_stats", player_stats, season)
        save("player_ratings", player_ratings, season)
        save("team_ratings", team_ratings, season)
    for season in SEASONS:
        save("game_logs", game_logs, season)
        save("games", games, season)
    # Lo de la temporada actual cambia a diario: se vuelve a bajar siempre
    for name, download in [
        ("game_logs", game_logs), ("games", games), ("player_ratings", player_ratings),
        ("schedule", schedule), ("rosters", rosters),
    ]:
        save(name, download, CURRENT_SEASON, refresh=True)


if __name__ == "__main__":
    main()
