import datetime as dt

import joblib
import pandas as pd
import streamlit as st

from data import CURRENT_SEASON, SEASONS, STAT_COLS, current_rosters, load, prev_season
from model import MODEL_PATH, contributions, lineup_stats, stats_table, weights

LABELS = {
    "PTS": "Puntos", "REB": "Rebotes", "AST": "Asistencias", "STL": "Robos", "BLK": "Tapones",
    "TOV": "Pérdidas", "MIN": "Minutos", "FG_PCT": "FG%", "FG3_PCT": "3P%", "PLUS_MINUS": "+/-",
}

st.set_page_config(page_title="Predicción NBA", page_icon=":material/sports_basketball:")


@st.cache_data
def load_csv(name, season):
    return load(name, season)


@st.cache_data
def load_stats_table(season):
    return stats_table(season)


@st.cache_data
def load_rosters():
    return current_rosters()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Predicción NBA")
players_tab, prediction_tab = st.tabs(["Jugadores", "Predicción"])

with players_tab:
    old, new = SEASONS[-2:]
    stats = {s: load_csv("player_stats", s).set_index("PLAYER_ID") for s in (old, new)}
    names = pd.concat([stats[old]["PLAYER_NAME"], stats[new]["PLAYER_NAME"]])
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
    st.dataframe(table, column_config={
        old: st.column_config.NumberColumn(format="%.1f"),
        new: st.column_config.NumberColumn(format="%.1f"),
        "Diferencia": st.column_config.NumberColumn(format="%+.1f"),
    })
    st.bar_chart(table[[old, new]], horizontal=True, sort=False, stack=False)

with prediction_tab:
    schedule = load_csv("schedule", CURRENT_SEASON)
    rosters = load_rosters()
    ps, bench = load_stats_table(prev_season(CURRENT_SEASON))
    model = load_model()

    dates = pd.to_datetime(schedule["GAME_DATE"]).dt.date
    upcoming = dates[dates >= dt.date.today()]
    day = st.date_input(
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
    game_id = st.selectbox("Partido", matchups, format_func=matchups.get)
    game = day_games.loc[game_id]

    def pick_lineup(container, side, team_id, team):
        """Selector de 5 jugadores; por defecto los 5 con más minutos en la temporada anterior."""
        roster = rosters[rosters["TEAM_ID"] == team_id].copy()
        roster["MIN"] = roster["PLAYER_ID"].map(ps["MIN"])
        roster = roster.sort_values("MIN", ascending=False)
        labels = dict(zip(roster["PLAYER_ID"], roster["PLAYER_NAME"] + roster["MIN"].isna().map({True: " (sin stats)", False: ""})))
        return container.multiselect(
            f"{side}: {team}",
            labels,
            default=list(labels)[:5],
            format_func=labels.get,
            max_selections=5,
            key=f"{game_id}_{team_id}",
        )

    away_col, home_col = st.columns(2)
    away_ids = pick_lineup(away_col, "Visitante", game.AWAY_TEAM_ID, game.AWAY_TEAM)
    home_ids = pick_lineup(home_col, "Local", game.HOME_TEAM_ID, game.HOME_TEAM)

    if len(away_ids) < 5 or len(home_ids) < 5:
        st.warning("Elige 5 jugadores por equipo.")
    else:
        home_avg = lineup_stats(home_ids, ps, bench)
        away_avg = lineup_stats(away_ids, ps, bench)
        features = home_avg - away_avg
        p_home = model.predict_proba(features.to_frame().T)[0, 1]
        away_col.metric(f"Victoria {game.AWAY_TEAM}", f"{1 - p_home:.0%}", border=True)
        home_col.metric(f"Victoria {game.HOME_TEAM}", f"{p_home:.0%}", border=True)

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
            f"Cada stat compara el promedio de las dos alineaciones (stats de {prev_season(CURRENT_SEASON)}). "
            f"Ventaja de local {home_edge:+.2f} + alineaciones {lineups:+.2f} = {home_edge + lineups:+.2f} "
            f"→ {p_home:.0%} para {game.HOME_TEAM}."
        )

    st.caption(
        f"Según las stats de {prev_season(CURRENT_SEASON)}. "
        "Los jugadores sin stats cuentan como un jugador de banca promedio."
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
