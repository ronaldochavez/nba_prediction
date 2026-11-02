"""Ficha de cada partido: los datos para contar su historia, como los avances de las cuentas de basquetbol.

Junta quién defendió a las estrellas la temporada pasada (y si esos defensores siguen disponibles), qué
pasó en los partidos sin los lesionados, el cansancio de cada equipo y el historial entre los dos.
Todas las funciones reciben los datos ya cargados y devuelven tablas listas para mostrar.
"""
import numpy as np
import pandas as pd


def star_defenders(star_id, opp_team_id, matchups, names, status):
    """Defensores del rival que marcaron a una estrella: posesiones, puntos que le permitieron y tiros.

    `status` dice dónde está hoy cada defensor («Disponible», «Fuera (Knee)», «Ahora en ATL»...).
    """
    rows = matchups[(matchups["OFF_PLAYER_ID"] == star_id) & (matchups["DEF_TEAM_ID"] == opp_team_id)]
    rows = rows.sort_values("PARTIAL_POSS", ascending=False)
    return pd.DataFrame({
        "PLAYER_ID": rows["DEF_PLAYER_ID"],
        "Defensor": rows["DEF_PLAYER_ID"].map(names).fillna(rows["DEF_PLAYER_ID"].astype(str)),
        "PJ": rows["GP"],
        "Posesiones": rows["PARTIAL_POSS"],
        "Puntos": rows["PLAYER_PTS"],
        "Tiros": rows["MATCHUP_FGM"].astype(int).astype(str) + "-" + rows["MATCHUP_FGA"].astype(int).astype(str),
        "Hoy": rows["DEF_PLAYER_ID"].map(status),
    }).reset_index(drop=True)


def absence_split(team_id, absent_ids, logs):
    """Partidos de la temporada sin los ausentes y cuánto anotó cada compañero en ellos contra con ellos.

    Devuelve los partidos sin ellos y una tabla de compañeros ordenada por cuánto subieron.
    """
    team_logs = logs[logs["TEAM_ID"] == team_id]
    played = team_logs.groupby("GAME_ID")["PLAYER_ID"].apply(set)
    without = [g for g, ids in played.items() if not ids & set(absent_ids)]
    if not without or len(without) == len(played):
        return without, pd.DataFrame()
    others = team_logs[~team_logs["PLAYER_ID"].isin(absent_ids)]
    split = others.assign(SIN=others["GAME_ID"].isin(without)).groupby(["PLAYER_ID", "SIN"]).agg(
        PJ=("PTS", "size"), PTS=("PTS", "mean"), MIN=("MIN", "mean"), NAME=("PLAYER_NAME", "last")
    ).unstack("SIN").dropna()
    table = pd.DataFrame({
        "Jugador": split[("NAME", True)],
        "PJ sin": split[("PJ", True)].astype(int),
        "Pts sin": split[("PTS", True)],
        "Pts con": split[("PTS", False)],
        "Min sin": split[("MIN", True)],
        "Min con": split[("MIN", False)],
    })
    table["Diferencia"] = table["Pts sin"] - table["Pts con"]
    return without, table.sort_values("Diferencia", ascending=False).reset_index()


def last_game(team_id, day, logs, team_abbr):
    """Último partido del equipo antes de `day`: fecha, rival, resultado y los que jugaron 30 minutos o más."""
    team_logs = logs[(logs["TEAM_ID"] == team_id) & (pd.to_datetime(logs["GAME_DATE"]).dt.date < day)]
    if team_logs.empty:
        return None
    game_id = team_logs.sort_values("GAME_DATE")["GAME_ID"].iloc[-1]
    game = team_logs[team_logs["GAME_ID"] == game_id]
    date = pd.to_datetime(game["GAME_DATE"].iloc[0]).date()
    heavy = game[game["MIN"] >= 30].sort_values("MIN", ascending=False)
    return {
        "fecha": date,
        "dias": (day - date).days,
        "partido": game["MATCHUP"].iloc[0].replace(team_abbr, "").strip(),
        "resultado": "ganó" if game["WL"].iloc[0] == "W" else "perdió",
        "minutos": ", ".join(f"{r.PLAYER_NAME} {r.MIN:.0f}" for r in heavy.itertuples()),
    }


def meetings(team_a, team_b, logs, games, season):
    """Partidos de temporada regular entre los dos equipos con el marcador y los dos mejores anotadores de cada uno.

    Marca los que se fueron a tiempo extra (más de 240 minutos de equipo) y las palizas (20 puntos o más).
    """
    pair = {team_a, team_b}
    rows = games[games[["HOME_TEAM_ID", "AWAY_TEAM_ID"]].apply(set, axis=1) == pair].sort_values("GAME_DATE")
    out = []
    for g in rows.itertuples():
        box = logs[logs["GAME_ID"] == g.GAME_ID]
        if box.empty:
            continue
        minutes = box.groupby("TEAM_ID")["MIN"].sum().max()
        margin = abs(g.HOME_PTS - g.AWAY_PTS)
        notes = (["tiempo extra"] if minutes > 242 else []) + (["paliza"] if margin >= 20 else [])
        scorers = []
        for team_id, abbr in [(g.AWAY_TEAM_ID, g.AWAY_TEAM), (g.HOME_TEAM_ID, g.HOME_TEAM)]:
            top = box[box["TEAM_ID"] == team_id].nlargest(2, "PTS")
            scorers.append(f"{abbr}: " + ", ".join(f"{r.PLAYER_NAME} {r.PTS:.0f} ({r.MIN:.0f} min)" for r in top.itertuples()))
        out.append({
            "Temporada": season,
            "Fecha": pd.to_datetime(g.GAME_DATE).strftime("%d/%m/%y"),
            "Partido": f"{g.AWAY_TEAM} {g.AWAY_PTS:.0f} @ {g.HOME_TEAM} {g.HOME_PTS:.0f}",
            "Notas": ", ".join(notes),
            "Mejores anotadores": " · ".join(scorers),
        })
    return pd.DataFrame(out)


def record(team_id, games):
    """Récord de la temporada y los últimos 5 resultados (G/P, del más viejo al más reciente)."""
    team = games[(games["HOME_TEAM_ID"] == team_id) | (games["AWAY_TEAM_ID"] == team_id)].sort_values("GAME_DATE")
    won = np.where(team["HOME_TEAM_ID"] == team_id, team["HOME_WIN"] == 1, team["HOME_WIN"] == 0)
    return f"{won.sum()}-{(~won).sum()}", "".join("G" if w else "P" for w in won[-5:])
