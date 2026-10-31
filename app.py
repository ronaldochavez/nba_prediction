import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from betting import (
    KELLY_FRACTIONS, american, decimal_odds, kelly, load_odds, market_summary, odds_results, odds_text, save_odds,
)
from data import CURRENT_SEASON, DATA_DIR, SEASONS, STAT_COLS, current_rosters, load, prev_season, refresh_current
from model import (
    BENCH_COLS, BENCH_SIZE, ELO_MEAN, K_GAMES, MODEL_PATH, back_to_back, contributions, current_stats, elo_ratings,
    expected_minutes, game_features, regular_season_logs, team_features, weights,
)
from points import current_ratings, lineup_ratings, project
from tracking import projected_margin, season_tracking, summary

LABELS = {
    "PTS": "Puntos", "REB": "Rebotes", "AST": "Asistencias", "STL": "Robos", "BLK": "Tapones",
    "TOV": "Pérdidas", "MIN": "Minutos", "FG_PCT": "FG%", "FG3_PCT": "3P%", "PLUS_MINUS": "+/-",
}
FEATURE_LABELS = {
    **LABELS, "BENCH_PTS": "Banca: puntos", "BENCH_PLUS_MINUS": "Banca: +/-", "ELO": "Elo",
    "B2B_HOME": "Local jugó ayer", "B2B_AWAY": "Visitante jugó ayer",
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


@st.cache_data
def load_game_context():
    """Elo actual de cada equipo y si cada equipo de cada partido del calendario jugó el día anterior."""
    schedule = load("schedule", CURRENT_SEASON)
    b2b_home, b2b_away = back_to_back(schedule)
    return elo_ratings()[1], pd.DataFrame({"HOME": b2b_home, "AWAY": b2b_away}).set_axis(schedule["GAME_ID"])


@st.cache_resource
def load_model(modified):
    """`modified` es la fecha del archivo: al reentrenar con model.py la app carga el modelo nuevo sola."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_tracking(modified):
    return season_tracking(CURRENT_SEASON, load_model(modified))


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
    """Promedio y partidos de la última temporada que jugó (2025-26, o 2024-25 si se perdió esa)."""
    for s in (new, old):
        if player_id in stats[s].index:
            return stats[s].loc[player_id, STAT_COLS + ["GP"]].astype(float)
    return pd.Series(np.nan, index=STAT_COLS + ["GP"])


def face_to_face(lineup_ids, roster_ids, labels, view, per36):
    """Stats clave de todo el equipo en la vista elegida: la alineación con su promedio y la banca con el suyo.

    El promedio de la alineación es simple, como en el modelo; el de la banca pondera por minutos jugados
    para que un jugador con 2 minutos no pese igual que uno con 20.
    """
    rows = []
    for pid in lineup_ids + [p for p in roster_ids if p not in lineup_ids]:
        base = season_average(pid)
        base = scale_36(base) if per36 else base
        if view in FORM_VIEWS:
            games = recent_games(pid, FORM_VIEWS[view])
            avg = averages(games, per36) if len(games) else pd.Series(np.nan, index=STAT_COLS)
            rows.append({
                "Jugador": labels[pid], "PJ": len(games), **avg[list(FACE_COLS)].to_dict(),
                "Tend.": avg["PTS"] - base["PTS"],
            })
        else:
            rows.append({"Jugador": labels[pid], "PJ": base["GP"], **base[list(FACE_COLS)].to_dict()})
    def weighted(col, w):
        return (col * w).sum() / (w[col.notna()].sum() or np.nan)

    table = pd.DataFrame(rows).set_index("Jugador")
    lineup, bench_rows = table.iloc[:len(lineup_ids)], table.iloc[len(lineup_ids):]
    played = (bench_rows["MIN"] * bench_rows["PJ"]).fillna(0)
    bench_avg = bench_rows.apply(weighted, w=played)
    bench_avg["MIN"] = weighted(bench_rows["MIN"], bench_rows["PJ"].fillna(0))  # minutos por partido jugado
    table = pd.concat([
        lineup, lineup.mean().to_frame("Promedio alineación").T,
        bench_rows, bench_avg.to_frame("Promedio banca").T,
    ])
    table.loc[["Promedio alineación", "Promedio banca"], "PJ"] = np.nan
    table["FG_PCT"] *= 100
    return table.rename(columns=FACE_COLS)


schedule = load_csv("schedule", CURRENT_SEASON)
finished = load_csv("games", CURRENT_SEASON).set_index("GAME_ID")
rosters = load_rosters()
ps, bench = load_current_stats()
minutes = load_expected_minutes()
elo, b2b_table = load_game_context()
injuries = load_csv("injuries", CURRENT_SEASON).set_index("PLAYER_ID")
model_modified = MODEL_PATH.stat().st_mtime
model = load_model(model_modified)
last_ids = set(load_csv("player_stats", LAST)["PLAYER_ID"])
older_ids = set(load_csv("player_stats", OLDER)["PLAYER_ID"])
current_ids = set(regular_season_logs(CURRENT_SEASON)["PLAYER_ID"])
dates = pd.to_datetime(schedule["GAME_DATE"]).dt.date
# El calendario usa la fecha de Nueva York; así no depende de la zona horaria del servidor
today = pd.Timestamp.now(tz="America/New_York").date()
upcoming = dates[dates >= today]
first_day = upcoming.min() if len(upcoming) else dates.min()


def tag(player_id):
    if player_id in last_ids:
        return ""
    if player_id in older_ids:
        return f" (stats {OLDER})"
    if player_id in current_ids:
        return f" (solo {CURRENT_SEASON})"
    return " (sin stats)"


def injury_status(player_id, day):
    """«fuera», «duda» o "" para un partido en `day`, según el reporte de lesiones.

    Para partidos de hoy o anteriores cuenta el reporte tal cual; para partidos futuros, un jugador
    fuera vuelve a estar disponible desde su fecha de regreso estimada.
    """
    if player_id not in injuries.index:
        return ""
    row = injuries.loc[player_id]
    if row.STATUS == "Day-To-Day":
        return "duda"
    back = pd.to_datetime(row.RETURN_DATE).date() if pd.notna(row.RETURN_DATE) else None
    return "" if day > today and back and back <= day else "fuera"


def injury_note(roster):
    """Lesionados de una plantilla: «Fuera: Nombre (Knee, vuelve ~20/11). En duda: Nombre (Ankle).»"""
    groups = []
    for label, status in [("Fuera", "fuera"), ("En duda", "duda")]:
        names = []
        for pid, name in roster.loc[roster["STATUS"] == status, ["PLAYER_ID", "PLAYER_NAME"]].itertuples(index=False):
            info = injuries.loc[pid]
            detail = [info.INJURY] if pd.notna(info.INJURY) else []
            if status == "fuera" and pd.notna(info.RETURN_DATE):
                detail.append(f"vuelve ~{pd.to_datetime(info.RETURN_DATE):%d/%m}")
            names.append(name + (f" ({', '.join(detail)})" if detail else ""))
        if names:
            groups.append(f"{label}: {', '.join(names)}")
    return ". ".join(groups) + "." if groups else ""


def pick_lineup(container, side, team_id, team, key, day):
    """Selector de 5 jugadores; por defecto los 5 con más minutos esperados que no están fuera por lesión.

    Devuelve los elegidos, su banca (los BENCH_SIZE siguientes disponibles), la plantilla disponible de más a
    menos minutos esperados y el nombre de cada jugador.
    """
    roster = rosters[rosters["TEAM_ID"] == team_id].copy()
    roster["EXPECTED_MIN"] = roster["PLAYER_ID"].map(minutes).fillna(0)
    roster = roster.sort_values("EXPECTED_MIN", ascending=False)
    roster["STATUS"] = [injury_status(pid, day) for pid in roster["PLAYER_ID"]]
    labels = {
        pid: name + tag(pid) + (f" ({status})" if status else "")
        for pid, name, status in zip(roster["PLAYER_ID"], roster["PLAYER_NAME"], roster["STATUS"])
    }
    available = list(roster.loc[roster["STATUS"] != "fuera", "PLAYER_ID"])
    ids = container.multiselect(
        f"{side}: {team}",
        labels,
        default=available[:5],
        format_func=labels.get,
        max_selections=5,
        key=key,
    )
    if note := injury_note(roster):
        container.caption(note)
    return ids, [p for p in available if p not in ids][:BENCH_SIZE], available, labels


def team_elo(team_id):
    return elo.get(team_id, ELO_MEAN)  # partidos de la Copa NBA aún sin equipos


def game_prediction(game_id, game, home, away):
    """Probabilidad de que gane el local; `home` y `away` son (alineación, banca) de cada equipo.

    También devuelve las features del partido y las stats de cada equipo para explicarla.
    """
    home_f, away_f = team_features(*home, ps, bench), team_features(*away, ps, bench)
    b2b = b2b_table.loc[game_id]
    features = game_features(
        home_f, away_f, team_elo(game.HOME_TEAM_ID) - team_elo(game.AWAY_TEAM_ID), b2b["HOME"], b2b["AWAY"]
    )
    return model.predict_proba(features.to_frame().T)[0, 1], features, home_f, away_f


def tired_teams(game_id, game):
    """Equipos del partido que jugaron el día anterior."""
    return [team for team, side in [(game.AWAY_TEAM, "AWAY"), (game.HOME_TEAM, "HOME")] if b2b_table.loc[game_id, side]]


def slate_game(game_id, game, bank, fraction, saved_odds):
    """Un partido de la jornada: alineaciones, probabilidad, momios y apuesta sugerida.

    Los momios válidos se guardan con la probabilidad del modelo mientras el partido no haya terminado.
    Devuelve la apuesta elegida, o None si no hay, para el resumen del día.
    """
    day = pd.Timestamp(game.GAME_DATE).date()
    header = f"**{game.AWAY_TEAM} @ {game.HOME_TEAM}**"
    if game_id in finished.index:
        result = finished.loc[game_id]
        header += f" · Final: {result.AWAY_PTS:.0f}-{result.HOME_PTS:.0f}"
    if tired := tired_teams(game_id, game):
        header += f" · Jugó ayer: {', '.join(tired)}"
    st.markdown(header)
    away_col, home_col = st.columns(2)
    away_ids, away_bench, _, _ = pick_lineup(
        away_col, "Visitante", game.AWAY_TEAM_ID, game.AWAY_TEAM, f"slate_{game_id}_{game.AWAY_TEAM_ID}", day
    )
    home_ids, home_bench, _, _ = pick_lineup(
        home_col, "Local", game.HOME_TEAM_ID, game.HOME_TEAM, f"slate_{game_id}_{game.HOME_TEAM_ID}", day
    )
    if len(away_ids) < 5 or len(home_ids) < 5:
        st.warning("Elige 5 jugadores por equipo.")
        return None

    p_home = game_prediction(game_id, game, (home_ids, home_bench), (away_ids, away_bench))[0]
    probs, odds = {game.AWAY_TEAM: 1 - p_home, game.HOME_TEAM: p_home}, {}
    # Fila aparte para que las tarjetas queden parejas aunque una alineación ocupe más renglones
    for col, (team, p) in zip(st.columns(2), probs.items()):
        col.metric(
            f"Victoria {team}", f"{p:.0%}", f"Momio justo {american(1 / p):+.0f}",
            delta_color="off", delta_arrow="off", border=True,
        )
        saved = saved_odds.get((game_id, team))
        text = col.text_input(
            f"Momio {team}", value=odds_text(saved[0]) if saved else "", key=f"slate_{game_id}_{team}_odds",
            placeholder="-110, +150 o 1.91",
        )
        if not text:
            continue
        odds[team] = decimal_odds(text)
        if odds[team] is None:
            col.error("Momio no válido: escribe algo como -110, +150 o 1.91.")
            continue
        col.caption(f"El momio implica {1 / odds[team]:.0%}: ventaja de {(p - 1 / odds[team]) * 100:+.1f} puntos")
        changed = saved is None or abs(saved[0] - odds[team]) > 1e-9 or abs(saved[1] - p) > 1e-4
        if changed and game_id not in finished.index:
            save_odds(game_id, game.GAME_DATE, team, odds[team], p)

    pick = st.segmented_control("Apostar a", list(probs), key=f"slate_{game_id}_pick")
    if pick is None:
        return None
    if odds.get(pick) is None:
        st.info(f"Escribe un momio válido para {pick}.")
        return None
    p, d = probs[pick], odds[pick]
    share = kelly(p, d) * fraction
    if share == 0:
        st.warning(f"Sin ventaja: el modelo le da {p:.0%} a {pick} y el momio implica {1 / d:.0%}. Mejor no apostar.")
        return None
    bet = {
        "Partido": f"{game.AWAY_TEAM} @ {game.HOME_TEAM}", "Equipo": pick, "Momio": american(d), "Prob. modelo": p * 100,
        "% del bank": share * 100,
    }
    if bank:
        bet.update({
            "Apuesta": bank * share, "Ganancia si gana": bank * share * (d - 1), "Valor esperado": bank * share * (p * d - 1),
        })
        st.success(  # \$ para que Streamlit no lea el texto entre dos $ como fórmula
            f"Apuesta **\\${bet['Apuesta']:,.2f}** a {pick} ({share:.1%} del bank). "
            f"Si gana: +\\${bet['Ganancia si gana']:,.2f}."
        )
    else:
        st.success(f"Apuesta **{share:.1%} de tu bank** a {pick}. Escribe tu bank arriba para ver la cantidad.")
    return bet


def refresh_data():
    """Descarga lo de la temporada actual y limpia la caché para que la app lea los datos nuevos.

    Devuelve cuántos partidos terminados nuevos hay.
    """
    before = len(finished)
    refresh_current()
    st.cache_data.clear()
    return len(load("games", CURRENT_SEASON)) - before


title_col, update_col = st.columns([3, 1], vertical_alignment="bottom")
title_col.title("Predicción NBA")
with update_col:
    if st.button(
        "Actualizar datos",
        icon=":material/refresh:",
        width="stretch",
        help=f"Descarga de stats.nba.com los resultados, box scores, ratings, plantillas y calendario de {CURRENT_SEASON}, "
        "y de ESPN el reporte de lesiones. Tarda alrededor de medio minuto.",
    ):
        with st.spinner("Descargando de stats.nba.com y ESPN..."):
            try:
                st.session_state["new_games"] = refresh_data()
            except Exception as e:
                st.error(f"No se pudo actualizar: {e}")
            else:
                st.rerun()
    last_game = pd.to_datetime(finished["GAME_DATE"]).max()
    downloaded = dt.datetime.fromtimestamp((DATA_DIR / f"games_{CURRENT_SEASON}.csv").stat().st_mtime)
    st.caption(
        (f"Último partido: {last_game:%d/%m} · " if pd.notna(last_game) else "")
        + f"descargado el {downloaded:%d/%m %H:%M}"
    )
if "new_games" in st.session_state:
    n = st.session_state.pop("new_games")
    st.toast(
        f"Datos actualizados: {n} partidos nuevos." if n else "Datos actualizados: no hay partidos nuevos.",
        icon=":material/check:",
    )

players_tab, prediction_tab, slate_tab, tracking_tab = st.tabs(["Jugadores", "Predicción", "Jornada", "Seguimiento"])

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

# Van antes que Predicción en el código porque esa pestaña puede cortar la ejecución con st.stop()
with tracking_tab:
    track = load_tracking(model_modified)
    if track.empty:
        st.info(f"Todavía no hay partidos terminados de {CURRENT_SEASON}.")
    else:
        s = summary(track)
        with st.container(horizontal=True):
            st.metric("Partidos", s["Partidos"], border=True)
            st.metric("Ganador acertado", f"{s['Aciertos del ganador']:.0%}", border=True)
            st.metric("Error de puntos por equipo", f"{s['TEAM por equipo']:.1f}", border=True)
            st.metric("Error del total", f"{s['TEAM total']:.1f}", border=True)
            st.metric(
                "Diferencia cumplida", f"{s['TEAM diferencia']:.0%}", border=True,
                help="Partidos en que el equipo que el marcador proyectado (por equipo) ponía arriba ganó por al menos "
                "esa diferencia",
            )

        hit = (track["P_HOME"] >= 0.5) == (track["HOME_WIN"] == 1)
        score = lambda away, home: away.round().astype(int).astype(str) + "-" + home.round().astype(int).astype(str)
        favorite, margin, real = projected_margin(track, "TEAM")
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
                "Dif. proy.": favorite + " por " + margin.round(1).astype(str),
                "Dif. real": real,
                "Cumplió": real >= margin,
                "Total": track["AWAY_PTS"] + track["HOME_PTS"],
                "Total proy.": track["TEAM_AWAY"] + track["TEAM_HOME"],
            }),
            hide_index=True,
            column_config={
                "Prob. local": st.column_config.NumberColumn(format="%.0f%%"),
                "Acierto": st.column_config.CheckboxColumn(),
                "Dif. proy.": st.column_config.TextColumn(
                    help="Equipo que el marcador proyectado por equipo pone arriba y por cuántos puntos"
                ),
                "Dif. real": st.column_config.NumberColumn(
                    format="%+.0f", help="Por cuánto ganó (+) o perdió (−) ese mismo equipo"
                ),
                "Cumplió": st.column_config.CheckboxColumn(help="Ganó por al menos la diferencia proyectada"),
                "Total proy.": st.column_config.NumberColumn(format="%.0f", help="Proyección por equipo"),
            },
        )
        st.caption(
            "Cada predicción usa solo lo que se sabía antes del partido. La alineación son los 5 con más minutos "
            "esperados entre los que jugaron, como si se hubiera revisado el reporte de lesiones. Los errores de arriba "
            f"son de la proyección por equipo; por alineación: {s['LINEUP por equipo']:.1f} por equipo y "
            f"{s['LINEUP total']:.1f} en el total (usa los ratings de jugadores de {LAST}, sin mezclar con "
            f"{CURRENT_SEASON}). Con pocos partidos el porcentaje de aciertos varía mucho: en las temporadas 2023-24 "
            "a 2025-26 fue de 68.4 %. "
            f"La diferencia proyectada por alineación se cumplió en {s['LINEUP diferencia']:.0%} de los partidos; en "
            "2023-24 a 2025-26, 58 % por equipo (67 % con diferencias de 6 a 10 puntos) y 55 % por alineación. "
            "Las diferencias proyectadas son chicas (de 3 a 3.5 puntos en promedio) porque la fórmula promedia el ataque "
            "de un equipo con la defensa del otro, así que no equivalen al hándicap de una casa de apuestas."
        )

    st.subheader("Modelo contra el mercado")
    results = odds_results(load_odds(), finished.reset_index())
    if results.empty:
        st.info(
            "Todavía no hay momios guardados de partidos terminados. Los momios que escribes en la pestaña Jornada "
            "antes de cada partido se guardan solos, y aquí verás si el modelo le gana a la casa."
        )
    else:
        m = market_summary(results)
        with st.container(horizontal=True):
            st.metric("Partidos con los dos momios", m["Partidos con los dos momios"], border=True)
            if m["Partidos con los dos momios"]:
                st.metric("Aciertos modelo / mercado", f"{m['Aciertos modelo']:.0%} / {m['Aciertos mercado']:.0%}", border=True)
                st.metric(
                    "Brier modelo / mercado", f"{m['Brier modelo']:.3f} / {m['Brier mercado']:.3f}", border=True,
                    help="Error medio al cuadrado de las probabilidades (0 = perfecto, 0.25 = siempre 50 %). "
                    "Menos es mejor: si el del modelo es más bajo, sus probabilidades son mejores que las de la casa.",
                )
            st.metric("Apuestas con ventaja", m["Apuestas con ventaja"], border=True)
            if m["Apuestas con ventaja"]:
                st.metric(
                    "Ganancia por unidad", f"{m['Ganancia por unidad']:+.1%}", border=True,
                    help="Lo que habría dejado apostar 1 unidad en cada momio donde el modelo veía ventaja",
                )
        results = results.sort_values(["GAME_DATE", "GAME_ID"], ascending=False)
        st.dataframe(
            pd.DataFrame({
                "Fecha": pd.to_datetime(results["GAME_DATE"]).dt.strftime("%d/%m/%y"),
                "Partido": results["AWAY_TEAM"] + " @ " + results["HOME_TEAM"],
                "Equipo": results["TEAM"],
                "Momio": results["ODDS"].map(american),
                "Prob. mercado": results["MARKET_P"] * 100,
                "Prob. modelo": results["MODEL_P"] * 100,
                "Ventaja": results["EDGE"] * 100,
                "Ganó": results["WON"],
                "1 unidad": results["PROFIT"],
            }),
            hide_index=True,
            column_config={
                "Momio": st.column_config.NumberColumn(format="%+.0f"),
                "Prob. mercado": st.column_config.NumberColumn(
                    format="%.0f%%", help="Sin el margen de la casa cuando están los momios de los dos equipos"
                ),
                "Prob. modelo": st.column_config.NumberColumn(format="%.0f%%"),
                "Ventaja": st.column_config.NumberColumn(
                    format="%+.1f", help="Probabilidad del modelo − probabilidad que implica el momio, en puntos"
                ),
                "Ganó": st.column_config.CheckboxColumn(),
                "1 unidad": st.column_config.NumberColumn(
                    format="%+.2f", help="Resultado de apostar 1 unidad si había ventaja; vacío si no la había"
                ),
            },
        )
        st.caption(
            "Se guarda el último momio que escribiste antes de que terminara el partido y la probabilidad que daba el "
            "modelo en ese momento, con las alineaciones elegidas. Hacen falta cientos de apuestas para que la ganancia "
            "por unidad diga algo: con pocas, depende sobre todo de la suerte. El archivo "
            f"data/odds_{CURRENT_SEASON}.csv se queda en tu computadora y no se sube a GitHub."
        )

with slate_tab:
    day_col, bank_col, kelly_col = st.columns(3, vertical_alignment="bottom")
    slate_day = day_col.date_input(
        "Fecha",
        value=first_day,
        min_value=dates.min(),
        max_value=dates.max(),
        format="DD/MM/YYYY",
        key="slate_day",
    )
    bank = bank_col.number_input("Bank ($)", min_value=0.0, value=None, step=100.0, placeholder="Ej. 1000", key="bank")
    fraction = kelly_col.segmented_control(
        "Fracción de Kelly",
        KELLY_FRACTIONS,
        default="1/4",
        required=True,
        key="kelly",
        help="Kelly completo es la apuesta que más hace crecer el bank si las probabilidades del modelo fueran exactas; "
        "como son estimaciones, 1/2 o 1/4 apuestan menos y aguantan mejor las rachas malas.",
    )
    slate = schedule[dates == slate_day].set_index("GAME_ID")
    if slate.empty:
        st.info("No hay partidos ese día.")
    else:
        day_summary = st.container()  # no se llama summary para no tapar la función de tracking
        bets = []
        saved_odds = {(r.GAME_ID, r.TEAM): (r.ODDS, r.MODEL_P) for r in load_odds().itertuples()}
        for game_id, game in slate.iterrows():
            with st.container(border=True):
                bet = slate_game(game_id, game, bank, KELLY_FRACTIONS[fraction], saved_odds)
            if bet:
                bets.append(bet)

        with day_summary:
            if not bets:
                st.caption(
                    "En cada partido elige a quién apostarle y escribe su momio; aquí aparece el resumen de las apuestas "
                    "del día."
                )
            else:
                bets = pd.DataFrame(bets)
                with st.container(horizontal=True):
                    st.metric("Apuestas", len(bets), border=True)
                    if bank:
                        st.metric("Total a apostar", f"${bets['Apuesta'].sum():,.2f}", border=True)
                        expected = bets["Valor esperado"].sum()
                        st.metric(
                            "Ganancia esperada", f"{'+' if expected >= 0 else '-'}${abs(expected):,.2f}", border=True,
                            help="Lo que ganarías en promedio si las probabilidades del modelo fueran exactas",
                        )
                    else:
                        st.metric("Total a apostar", f"{bets['% del bank'].sum():.1f}% del bank", border=True)
                st.dataframe(bets, hide_index=True, column_config={
                    "Momio": st.column_config.NumberColumn(format="%+.0f"),
                    "Prob. modelo": st.column_config.NumberColumn(format="%.0f%%"),
                    "% del bank": st.column_config.NumberColumn(format="%.1f%%"),
                    **{
                        c: st.column_config.NumberColumn(format="$%.2f")
                        for c in ["Apuesta", "Ganancia si gana", "Valor esperado"]
                    },
                })
                if bets["% del bank"].sum() > 100:
                    st.warning("Las apuestas suman más que tu bank: usa una fracción de Kelly más chica.")

        st.caption(
            "Momio justo = el que pagaría exactamente la probabilidad del modelo: si la casa paga más (por ejemplo +150 "
            "cuando el justo es +120), hay ventaja. La apuesta sugerida sigue el criterio de Kelly: "
            "(probabilidad × momio decimal − 1) / (momio decimal − 1) del bank, por la fracción elegida, y es 0 si no "
            "hay ventaja. Cada apuesta se calcula sobre el bank completo. Los momios que escribes se guardan para "
            "compararlos con el modelo en la pestaña Seguimiento. Las probabilidades son estimaciones: el modelo acertó "
            "al ganador en 68.4 % de los partidos de 2023-24 a 2025-26. Revisa que los 5 elegidos de verdad jueguen: "
            "la alineación sugerida saca a los que están fuera según ESPN, pero no a los que están en duda."
        )

with prediction_tab:
    team_r, player_r, bench_r, home_edge_pts = load_current_ratings()
    date_col, game_col = st.columns([1, 2])
    day = date_col.date_input(
        "Fecha",
        value=first_day,
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

    away_col, home_col = st.columns(2)
    away_ids, away_bench, away_available, away_labels = pick_lineup(
        away_col, "Visitante", game.AWAY_TEAM_ID, game.AWAY_TEAM, f"{game_id}_{game.AWAY_TEAM_ID}", day
    )
    home_ids, home_bench, home_available, home_labels = pick_lineup(
        home_col, "Local", game.HOME_TEAM_ID, game.HOME_TEAM, f"{game_id}_{game.HOME_TEAM_ID}", day
    )

    if len(away_ids) < 5 or len(home_ids) < 5:
        st.warning("Elige 5 jugadores por equipo.")
    else:
        p_home, features, home_f, away_f = game_prediction(
            game_id, game, (home_ids, home_bench), (away_ids, away_bench)
        )
        away_metric, home_metric = st.columns(2)  # fila aparte para que queden parejas
        away_metric.metric(f"Victoria {game.AWAY_TEAM}", f"{1 - p_home:.0%}", border=True)
        home_metric.metric(f"Victoria {game.HOME_TEAM}", f"{p_home:.0%}", border=True)
        tired = tired_teams(game_id, game)
        st.caption(
            f"Elo: {game.AWAY_TEAM} {team_elo(game.AWAY_TEAM_ID):.0f} · {game.HOME_TEAM} {team_elo(game.HOME_TEAM_ID):.0f}. "
            + (f"Jugó ayer: {', '.join(tired)}." if tired else "Ninguno de los dos jugó ayer.")
        )

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
            "PJ": st.column_config.NumberColumn(format="%.0f", width=40, help="Partidos en los que se basa cada fila"),
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
        for container, side, team, ids, available, labels in [
            (face_away, "Visitante", game.AWAY_TEAM, away_ids, away_available, away_labels),
            (face_home, "Local", game.HOME_TEAM, home_ids, home_available, home_labels),
        ]:
            table = face_to_face(ids, available, labels, view, face_per36)
            container.markdown(f"**{side}: {team}**")
            # Alto suficiente para ver a toda la plantilla sin scroll (35 px por fila más el encabezado)
            container.dataframe(table, column_config=face_config, height=35 * (len(table) + 1) + 3, placeholder="—")
        st.caption(
            "Arriba los 5 elegidos y su promedio; abajo el resto de la plantilla disponible (sin los que están fuera por "
            "lesión), de más a menos minutos esperados, y el promedio de la banca ponderado por minutos jugados. "
            f"«Temporada» es el promedio de {new} (o de {old} si el jugador no jugó esa). «Últimos 5» y «Últimos 10» "
            "incluyen pretemporada, temporada regular y playoffs; en pretemporada los titulares juegan menos minutos, "
            "así que conviene mirarlos por 36 minutos. PJ = partidos de la vista. "
            "Tend. = puntos en la vista − promedio de la temporada."
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
        contrib = contributions(model, features)
        fmt = lambda c, v: f"{v * 100:.1f}" if c.endswith("_PCT") else f"{v:.1f}"
        rows = {FEATURE_LABELS[c]: (fmt(c, away_f[c]), fmt(c, home_f[c]), contrib[c]) for c in STAT_COLS + BENCH_COLS}
        rows["Elo"] = (f"{team_elo(game.AWAY_TEAM_ID):.0f}", f"{team_elo(game.HOME_TEAM_ID):.0f}", contrib["ELO"])
        rows["Jugó ayer"] = (
            "Sí" if features["B2B_AWAY"] else "No", "Sí" if features["B2B_HOME"] else "No",
            contrib["B2B_HOME"] + contrib["B2B_AWAY"],
        )
        detail = pd.DataFrame.from_dict(rows, orient="index", columns=[game.AWAY_TEAM, game.HOME_TEAM, "Aporte"])
        detail = detail.sort_values("Aporte", key=abs, ascending=False)

        table_col, chart_col = st.columns(2)
        table_col.dataframe(detail, column_config={
            game.AWAY_TEAM: st.column_config.TextColumn(alignment="right"),
            game.HOME_TEAM: st.column_config.TextColumn(alignment="right"),
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
        total = detail["Aporte"].sum()
        st.caption(
            f"Las stats comparan el promedio de las dos alineaciones; la banca, a los {BENCH_SIZE} siguientes disponibles "
            "de cada equipo en minutos esperados. El Elo resume la fuerza del equipo según sus resultados. "
            f"Ventaja de local {home_edge:+.2f} + aportes {total:+.2f} = {home_edge + total:+.2f} "
            f"→ {p_home:.0%} para {game.HOME_TEAM}."
        )

    updated = pd.to_datetime(injuries["UPDATED"]).max()
    st.caption(
        f"Stats de {LAST} mezcladas con lo que va de {CURRENT_SEASON}: la temporada actual pesa más con cada "
        f"partido (a los {K_GAMES} partidos, mitad y mitad). La alineación sugerida son los 5 con más minutos "
        "esperados que no están fuera según el reporte de lesiones de ESPN"
        + (f" (última novedad: {updated:%d/%m})" if pd.notna(updated) else "")
        + "; los que están en duda se quedan, revisa antes del partido. "
        f"Quien no jugó {LAST} usa sus stats de {OLDER}; los jugadores sin stats cuentan como un jugador de banca promedio."
    )

    with st.expander("Cuánto pesa cada dato en el modelo"):
        model_weights = weights(model).rename(index=FEATURE_LABELS)
        model_weights["Importancia"] *= 100
        st.dataframe(model_weights, column_config={
            "Peso": st.column_config.NumberColumn(
                format="%+.2f", help="Cambio en el puntaje por cada desviación estándar de diferencia entre los equipos"
            ),
            "Importancia": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        })
        st.caption(
            f"Aprendidos con los partidos de {SEASONS[0]} a {SEASONS[-1]}. "
            f"Ventaja de local: {model[-1].intercept_[0]:+.2f}."
        )
