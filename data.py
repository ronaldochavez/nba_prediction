"""Descarga los datos de la NBA con nba_api y los guarda en CSV dentro de data/."""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashplayerstats,
    playergamelogs,
    scheduleleaguev2,
)

DATA_DIR = Path(__file__).parent / "data"
MOVES_PATH = Path(__file__).parent / "roster_moves.csv"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]  # temporadas con partidos para entrenar
CURRENT_SEASON = "2026-27"
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]
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


def player_games(season):
    """Una fila por jugador y partido: quién jugó y cuántos minutos."""
    df = fetch(
        playergamelogs.PlayerGameLogs,
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    return df[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]


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
    })


def games(season):
    df = season_games(season)
    df["HOME_WIN"] = (df["HOME_PTS"] > df["AWAY_PTS"]).astype(int)
    return df


def schedule(season):
    return season_games(season).drop(columns=["HOME_PTS", "AWAY_PTS"])


def rosters(season):
    df = fetch(commonallplayers.CommonAllPlayers, is_only_current_season=1, season=season)
    df = df[df["TEAM_ID"] != 0].rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})
    return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]


def save(name, download, season):
    path = DATA_DIR / f"{name}_{season}.csv"
    if path.exists():
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

    pending = moves[[team_of.get(pid, "") == team for pid, team in zip(moves["PLAYER_ID"], moves["FROM_TEAM"])]]
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
    # Cada partido se predice con stats de la temporada anterior, así que también hace falta 2023-24
    for season in [prev_season(SEASONS[0])] + SEASONS:
        save("player_stats", player_stats, season)
    for season in SEASONS:
        save("player_games", player_games, season)
        save("games", games, season)
    save("schedule", schedule, CURRENT_SEASON)
    save("rosters", rosters, CURRENT_SEASON)


if __name__ == "__main__":
    main()
