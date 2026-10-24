import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data import CURRENT_SEASON, SEASONS, STAT_COLS, current_rosters, load, prev_season
from model import (
    K_GAMES, MODEL_PATH, contributions, current_stats, expected_minutes, lineup_stats, regular_season_logs, weights,
)
from points import current_ratings, lineup_ratings, project
from tracking import season_tracking, summary

LABELS = {
    "PTS": "Puntos", "REB": "Rebotes", "AST": "Asistencias", "STL": "Robos", "BLK": "Tapones",
    "TOV": "Pérdidas", "MIN": "Minutos", "FG_PCT": "FG%", "FG3_PCT": "3P%", "PLUS_MINUS": "+/-",
}
COUNTS = [c for c in STAT_COLS if not c.endswith("_PCT") and c != "MIN"]  # stats que se escalan a 36 minutos
GAME_TYPES = {"Pre Season": "Pretemporada", "Regular Season": "Temporada regular", "Playoffs": "Playoffs"}
LAST, OLDER = prev_season(CURRENT_SEASON), prev_season(prev_season(CURRENT_SEASON))
# Stats del cara a cara, con nombres cortos para que las dos tablas entren lado a lado
FACE_COLS = {"MIN": "Min", "PTS": "Pts", "REB": "Reb", "AST": "Ast", "STL": "Rob", "PLUS_MINUS": "+/-", "FG_PCT": "FG%"}
FORM_VIEWS = {"Últimos 5": 5, "Últimos 10": 10}

st.set_page_config(page_title="Predicción NBA", page_icon=":material/sports_basketball:", layout="wide")


@st.cache_data
def load_csv(name, season):
    return load(name, season)


@st.cache_data
def load_current_stats():
    return current_stats(CURRENT_SEASON)


@st.cache_data
def load_current_ratings():
    return current_ratings(CURRENT_SEASON)


@st.cache_data
def load_expected_minutes():
    return expected_minutes(CURRENT_SEASON)


@st.cache_data
def load_rosters():
    return current_rosters()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_tracking():
    return season_tracking(CURRENT_SEASON, load_model())


def averages(games, per36=False):
    """Promedio por partido (o por 36 minutos); FG% y 3P% salen del total de tiros."""
    avg = games[[c for c in STAT_COLS if not c.endswith("_PCT")]].mean()
    if per36:
        avg[COUNTS] = games[COUNTS].sum() / games["MIN"].sum() * 36
    fga, fg3a = games["FGA"].sum(), games["FG3A"].sum()
    avg["FG_PCT"] = games["FGM"].sum() / fga if fga else None
    avg["FG3_PCT"] = games["FG3M"].sum() / fg3a if fg3a else None
    return avg[STAT_COLS].astype(float)


def scale_36(avg):
    """Pasa un promedio por partido a por 36 minutos."""
    avg = avg.copy()
    avg[COUNTS] = avg[COUNTS] / avg["MIN"] * 36
    return avg


old, new = SEASONS[-2:]
stats = {s: load_csv("player_stats", s).set_index("PLAYER_ID") for s in (old, new)}
logs = pd.concat([load_csv("game_logs", new), load_csv("game_logs", CURRENT_SEASON)], ignore_index=True)


def recent_games(player_id, n):
    """Últimos n partidos del jugador (pretemporada, temporada regular y playoffs) desde la temporada pasada."""
    return logs[logs["PLAYER_ID"] == player_id].sort_values("GAME_DATE", ascending=False).head(n)


def season_average(player_id):
    """Promedio de la última temporada que jugó (2025-26, o 2024-25 si se perdió esa)."""
    for s in (new, old):
        if player_id in stats[s].index:
            return stats[s].loc[player_id, STAT_COLS].astype(float)
    return pd.Series(np.nan, index=STAT_COLS)


def face_to_face(player_ids, labels, view, per36):
    """Stats clave de cada jugador de una alineación en la vista elegida, más el promedio de los 5.

    También devuelve los jugadores que tienen menos partidos que los que pide la vista.
    """
    rows, short_on_games = [], []
    for pid in player_ids:
        base = season_average(pid)
        base = scale_36(base) if per36 else base
        row = {"Jugador": labels[pid]}
        if view in FORM_VIEWS:
            games = recent_games(pid, FORM_VIEWS[view])
            avg = averages(games, per36) if len(games) else pd.Series(np.nan, index=STAT_COLS)
            if len(games) < FORM_VIEWS[view]:
                short_on_games.append(f"{labels[pid]}: {len(games)} de {FORM_VIEWS[view]}")
            row.update(avg[list(FACE_COLS)])
            row["Tend."] = avg["PTS"] - base["PTS"]
        else:
            row.update(base[list(FACE_COLS)])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Jugador")
    table.loc["Promedio"] = table.mean()
    table["FG_PCT"] *= 100
    return table.rename(columns=FACE_COLS), short_on_games


st.title("Predicción NBA")
players_tab, prediction_tab, tracking_tab = st.tabs(["Jugadores", "Predicción", "Seguimiento"])

with players_tab:
    names = pd.concat([stats[old]["PLAYER_NAME"], stats[new]["PLAYER_NAME"], logs.set_index("PLAYER_ID")["PLAYER_NAME"]])
    names = names[~names.index.duplicated(keep="last")].sort_values()
    top_scorer = stats[new]["PTS"].idxmax()
    player_id = st.selectbox("Jugador", names.index, index=names.index.get_loc(top_scorer), format_func=names.get)

    rows = {s: stats[s].reindex([player_id]).iloc[0] for s in (old, new)}
    st.caption(" · ".join(
        f"{s}: {r.TEAM_ABBREVIATION}, {r.GP:.0f} partidos" if pd.notna(r.GP) else f"{s}: no jugó"
        for s, r in rows.items()
    ))

    table = pd.DataFrame({s: r[STAT_COLS].astype(float) for s, r in rows.items()})
    table.loc[["FG_PCT", "FG3_PCT"]] *= 100
    table["Diferencia"] = table[new] - table[old]
    table.index = table.index.map(LABELS)
    if table[[old, new]].notna().any().any():
        table_col, chart_col = st.columns(2)
        table_col.dataframe(table, column_config={
            old: st.column_config.NumberColumn(format="%.1f"),
            new: st.column_config.NumberColumn(format="%.1f"),
            "Diferencia": st.column_config.NumberColumn(format="%+.1f"),
        })
        chart_col.bar_chart(table[[old, new]], horizontal=True, sort=False, stack=False)

    st.subheader("Forma reciente")
    per36 = st.toggle(
        "Por 36 minutos",
        help="Escala las stats a 36 minutos para comparar sin el efecto de jugar más o menos minutos",
        key="players_per36",
    )
    recent = recent_games(player_id, 10)
    if recent.empty:
        st.info(f"No tiene partidos desde {new}.")
    else:
        short = f"Últimos {min(5, len(recent))}"  # puede haber menos partidos que 5 o 10
        form = pd.DataFrame({short: averages(recent.head(5), per36)})
        if len(recent) > 5:
            form[f"Últimos {len(recent)}"] = averages(recent, per36)
        baseline = next((s for s in (new, old) if pd.notna(rows[s].GP)), None)
        if baseline:
            base = rows[baseline][STAT_COLS].astype(float)
            form[f"Promedio {baseline}"] = scale_36(base) if per36 else base
            form["Tendencia"] = form[short] - form[f"Promedio {baseline}"]
        form.loc[["FG_PCT", "FG3_PCT"]] *= 100
        form.index = form.index.map(LABELS)
        form_col, form_chart_col = st.columns(2)
        form_col.dataframe(form, column_config={
            **{c: st.column_config.NumberColumn(format="%.1f") for c in form.columns},
            "Tendencia": st.column_config.NumberColumn(format="%+.1f"),
        })

        games = recent.sort_values("GAME_DATE").assign(
            Fecha=lambda d: pd.to_datetime(d["GAME_DATE"]).dt.strftime("%d/%m/%y"),
            Tipo=lambda d: d["SEASON_TYPE"].map(GAME_TYPES),
        )
        points_label = "Puntos por 36 min" if per36 else "Puntos"
        chart = games.assign(**{points_label: games["PTS"] * 36 / games["MIN"].replace(0, np.nan) if per36 else games["PTS"]})
        form_chart_col.bar_chart(chart, x="Fecha", y=points_label, color="Tipo", sort=False)
        st.caption(
            f"Incluye pretemporada, temporada regular y playoffs desde {new}. "
            + (f"Tendencia = {short.lower()} − promedio de {baseline}. " if baseline else "")
            + (
                "Por 36 minutos: puntos, rebotes, etc. divididos por los minutos jugados y multiplicados por 36."
                if per36
                else "En pretemporada los titulares suelen jugar menos minutos, así que sus números bajan; "
                "la vista por 36 minutos corrige ese efecto."
            )
        )

        with st.expander("Partido a partido"):
            detail = games.assign(
                Resultado=games["WL"].map({"W": "G", "L": "P"}),
                FG=games["FGM"].astype(str) + "-" + games["FGA"].astype(str),
                **{"3P": games["FG3M"].astype(str) + "-" + games["FG3A"].astype(str)},
            ).rename(columns={"MATCHUP": "Partido", **LABELS})
            st.dataframe(
                detail.iloc[::-1][[
                    "Fecha", "Partido", "Tipo", "Resultado", "Minutos", "Puntos", "Rebotes", "Asistencias",
                    "Robos", "Tapones", "Pérdidas", "FG", "3P", "+/-",
                ]],
                hide_index=True,
                column_config={"Minutos": st.column_config.NumberColumn(format="%.0f")},
            )

# Va antes que Predicción en el código porque esa pestaña puede cortar la ejecución con st.stop()
with tracking_tab:
    track = load_tracking()
    if track.empty:
        st.info(f"Todavía no hay partidos terminados de {CURRENT_SEASON}.")
    else:
        s = summary(track)
        with st.container(horizontal=True):
            st.metric("Partidos", s["Partidos"], border=True)
            st.metric("Ganador acertado", f"{s['Aciertos del ganador']:.0%}", border=True)
            st.metric("Error de puntos por equipo", f"{s['TEAM por equipo']:.1f}", border=True)
            st.metric("Error del total", f"{s['TEAM total']:.1f}", border=True)

        hit = (track["P_HOME"] >= 0.5) == (track["HOME_WIN"] == 1)
        score = lambda away, home: away.round().astype(int).astype(str) + "-" + home.round().astype(int).astype(str)
        st.dataframe(
            pd.DataFrame({
                "Fecha": pd.to_datetime(track["GAME_DATE"]).dt.strftime("%d/%m/%y"),
                "Partido": track["AWAY_TEAM"] + " @ " + track["HOME_TEAM"],
                "Prob. local": track["P_HOME"] * 100,
                "Pronóstico": np.where(track["P_HOME"] >= 0.5, track["HOME_TEAM"], track["AWAY_TEAM"]),
                "Ganó": np.where(track["HOME_WIN"] == 1, track["HOME_TEAM"], track["AWAY_TEAM"]),
                "Acierto": hit,
                "Marcador": score(track["AWAY_PTS"], track["HOME_PTS"]),
                "Proy. por equipo": score(track["TEAM_AWAY"], track["TEAM_HOME"]),
                "Proy. por alineación": score(track["LINEUP_AWAY"], track["LINEUP_HOME"]),
                "Total": track["AWAY_PTS"] + track["HOME_PTS"],
                "Total proy.": track["TEAM_AWAY"] + track["TEAM_HOME"],
            }),
            hide_index=True,
            column_config={
                "Prob. local": st.column_config.NumberColumn(format="%.0f%%"),
                "Acierto": st.column_config.CheckboxColumn(),
                "Total proy.": st.column_config.NumberColumn(format="%.0f", help="Proyección por equipo"),
            },
        )
        st.caption(
            "Cada predicción usa solo lo que se sabía antes del partido y la alineación que de verdad jugó "
            "(los 5 con más minutos). Los errores de arriba son de la proyección por equipo; por alineación: "
            f"{s['LINEUP por equipo']:.1f} por equipo y {s['LINEUP total']:.1f} en el total (usa los ratings de "
            f"jugadores de {LAST}, sin mezclar con {CURRENT_SEASON}). Con pocos partidos el porcentaje de aciertos "
            "varía mucho: en las temporadas 2023-24 a 2025-26 fue de 68.9 %."
        )

with prediction_tab:
    schedule = load_csv("schedule", CURRENT_SEASON)
    rosters = load_rosters()
    ps, bench = load_current_stats()
    team_r, player_r, bench_r, home_edge_pts = load_current_ratings()
    minutes = load_expected_minutes()
    model = load_model()
    last_ids = set(load_csv("player_stats", LAST)["PLAYER_ID"])
    older_ids = set(load_csv("player_stats", OLDER)["PLAYER_ID"])
    current_ids = set(regular_season_logs(CURRENT_SEASON)["PLAYER_ID"])

    dates = pd.to_datetime(schedule["GAME_DATE"]).dt.date
    upcoming = dates[dates >= dt.date.today()]
    date_col, game_col = st.columns([1, 2])
    day = date_col.date_input(
        "Fecha",
        value=upcoming.min() if len(upcoming) else dates.min(),
        min_value=dates.min(),
        max_value=dates.max(),
        format="DD/MM/YYYY",
    )
    day_games = schedule[dates == day].set_index("GAME_ID")
    if day_games.empty:
        st.info("No hay partidos ese día.")
        st.stop()

    matchups = {gid: f"{g.AWAY_TEAM} @ {g.HOME_TEAM}" for gid, g in day_games.iterrows()}
    game_id = game_col.selectbox("Partido", matchups, format_func=matchups.get)
    game = day_games.loc[game_id]

    def tag(player_id):
        if player_id in last_ids:
            return ""
        if player_id in older_ids:
            return f" (stats {OLDER})"
        if player_id in current_ids:
            return f" (solo {CURRENT_SEASON})"
        return " (sin stats)"

    def pick_lineup(container, side, team_id, team):
        """Selector de 5 jugadores; por defecto los 5 con más minutos esperados."""
        roster = rosters[rosters["TEAM_ID"] == team_id].copy()
        roster["EXPECTED_MIN"] = roster["PLAYER_ID"].map(minutes).fillna(0)
        roster = roster.sort_values("EXPECTED_MIN", ascending=False)
        labels = {pid: name + tag(pid) for pid, name in zip(roster["PLAYER_ID"], roster["PLAYER_NAME"])}
        ids = container.multiselect(
            f"{side}: {team}",
            labels,
            default=list(labels)[:5],
            format_func=labels.get,
            max_selections=5,
            key=f"{game_id}_{team_id}",
        )
        return ids, labels

    away_col, home_col = st.columns(2)
    away_ids, away_labels = pick_lineup(away_col, "Visitante", game.AWAY_TEAM_ID, game.AWAY_TEAM)
    home_ids, home_labels = pick_lineup(home_col, "Local", game.HOME_TEAM_ID, game.HOME_TEAM)

    if len(away_ids) < 5 or len(home_ids) < 5:
        st.warning("Elige 5 jugadores por equipo.")
    else:
        home_avg = lineup_stats(home_ids, ps, bench)
        away_avg = lineup_stats(away_ids, ps, bench)
        features = home_avg - away_avg
        p_home = model.predict_proba(features.to_frame().T)[0, 1]
        away_col.metric(f"Victoria {game.AWAY_TEAM}", f"{1 - p_home:.0%}", border=True)
        home_col.metric(f"Victoria {game.HOME_TEAM}", f"{p_home:.0%}", border=True)

        st.subheader("Cara a cara")
        season_view = f"Temporada {new}"
        with st.container(horizontal=True, vertical_alignment="center"):
            view = st.segmented_control(
                "Ver", [season_view, *FORM_VIEWS], default="Últimos 5", required=True, key="face_view",
                label_visibility="collapsed",
            )
            face_per36 = st.toggle("Por 36 minutos", key="face_per36")
        face_config = {
            "_index": st.column_config.TextColumn(width=190),
            **{
                short: st.column_config.NumberColumn(format="%.1f", width=52, help=LABELS[c])
                for c, short in FACE_COLS.items()
            },
            "Tend.": st.column_config.NumberColumn(
                format="%+.1f",
                width=52,
                help=f"Tendencia: puntos en esta vista − promedio de la temporada ({new}, o {old} si no la jugó)",
            ),
        }
        face_away, face_home = st.columns(2)
        short_on_games = []
        for container, side, team, ids, labels in [
            (face_away, "Visitante", game.AWAY_TEAM, away_ids, away_labels),
            (face_home, "Local", game.HOME_TEAM, home_ids, home_labels),
        ]:
            table, short = face_to_face(ids, labels, view, face_per36)
            short_on_games += short
            container.markdown(f"**{side}: {team}**")
            container.dataframe(table, column_config=face_config)
        st.caption(
            f"«Temporada» es el promedio de {new} (o de {old} si el jugador no jugó esa). «Últimos 5» y «Últimos 10» "
            "incluyen pretemporada, temporada regular y playoffs; en pretemporada los titulares juegan menos minutos, "
            "así que conviene mirarlos por 36 minutos. Tend. = puntos en la vista − promedio de la temporada."
            + (f" Con menos partidos: {', '.join(short_on_games)}." if short_on_games else "")
        )

        st.subheader("Marcador proyectado")
        by_team = project(team_r.loc[game.HOME_TEAM_ID], team_r.loc[game.AWAY_TEAM_ID], home_edge_pts)
        by_lineup = project(
            lineup_ratings(home_ids, player_r, bench_r), lineup_ratings(away_ids, player_r, bench_r), home_edge_pts
        )
        score = pd.DataFrame(
            [[away, home, away + home] for home, away in (by_team, by_lineup)],
            index=["Por equipo", "Por alineación"],
            columns=[game.AWAY_TEAM, game.HOME_TEAM, "Total"],
        )
        st.dataframe(
            score, width="content", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in score.columns}
        )
        st.caption(
            "Puntos = ((ataque propio × ritmo propio) / 100 + (defensa rival × ritmo rival) / 100) / 2 ± ventaja de "
            f"local, con los ratings de {LAST} mezclados con lo que va de {CURRENT_SEASON}. "
            "«Por equipo» usa los ratings de cada equipo y no cambia con la alineación; «Por alineación» promedia a "
            "los 5 jugadores elegidos. En partidos pasados el error típico fue de unos 10 puntos por equipo y 15 en "
            "el total."
        )

        st.subheader("¿Por qué este porcentaje?")
        detail = pd.DataFrame({
            game.AWAY_TEAM: away_avg,
            game.HOME_TEAM: home_avg,
            "Aporte": contributions(model, features),
        })
        detail.loc[["FG_PCT", "FG3_PCT"], [game.AWAY_TEAM, game.HOME_TEAM]] *= 100
        detail = detail.rename(index=LABELS).sort_values("Aporte", key=abs, ascending=False)

        table_col, chart_col = st.columns(2)
        table_col.dataframe(detail, column_config={
            game.AWAY_TEAM: st.column_config.NumberColumn(format="%.1f"),
            game.HOME_TEAM: st.column_config.NumberColumn(format="%.1f"),
            "Aporte": st.column_config.NumberColumn(format="%+.2f"),
        })
        chart_col.bar_chart(
            pd.DataFrame({
                f"Local: {game.HOME_TEAM}": detail["Aporte"].clip(lower=0),
                f"Visitante: {game.AWAY_TEAM}": detail["Aporte"].clip(upper=0),
            }),
            horizontal=True,
            sort=False,
            stack=True,
        )
        home_edge = model[-1].intercept_[0]
        lineups = detail["Aporte"].sum()
        st.caption(
            "Cada stat compara el promedio de las dos alineaciones. "
            f"Ventaja de local {home_edge:+.2f} + alineaciones {lineups:+.2f} = {home_edge + lineups:+.2f} "
            f"→ {p_home:.0%} para {game.HOME_TEAM}."
        )

    st.caption(
        f"Stats de {LAST} mezcladas con lo que va de {CURRENT_SEASON}: la temporada actual pesa más con cada "
        f"partido (a los {K_GAMES} partidos, mitad y mitad). La alineación sugerida son los 5 con más minutos "
        f"esperados. Quien no jugó {LAST} usa sus stats de {OLDER}; los jugadores sin stats cuentan como un "
        "jugador de banca promedio."
    )

    with st.expander("Cuánto pesa cada stat en el modelo"):
        model_weights = weights(model).rename(index=LABELS)
        model_weights["Importancia"] *= 100
        st.dataframe(model_weights, column_config={
            "Peso": st.column_config.NumberColumn(
                format="%+.2f", help="Cambio en el puntaje por cada desviación estándar de diferencia entre alineaciones"
            ),
            "Importancia": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        })
        st.caption(
            f"Aprendidos con los partidos de {SEASONS[0]} a {SEASONS[-1]}. "
            f"Ventaja de local: {model[-1].intercept_[0]:+.2f}."
        )
