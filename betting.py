"""Momios, tamaño de apuesta con el criterio de Kelly y registro de los momios de la casa."""
import datetime as dt

import numpy as np
import pandas as pd

from data import CURRENT_SEASON, DATA_DIR

# Fracción de Kelly: la completa maximiza el crecimiento del bank solo si las probabilidades son exactas
KELLY_FRACTIONS = {"Completo": 1, "1/2": 0.5, "1/4": 0.25}
# Momios escritos en la pestaña Jornada; es un archivo personal, no se sube a GitHub
ODDS_PATH = DATA_DIR / f"odds_{CURRENT_SEASON}.csv"
ODDS_COLS = ["GAME_ID", "GAME_DATE", "TEAM", "ODDS", "MODEL_P", "SAVED_AT"]


def decimal_odds(text):
    """Momio escrito como americano (+150, -120) o decimal (2.50) → decimal; None si no es válido.

    Un momio americano nunca está entre -100 y +100, así que lo que cae en ese rango se lee como decimal.
    """
    try:
        x = float(text.strip())
    except ValueError:
        return None
    if x >= 100:
        return 1 + x / 100
    if x <= -100:
        return 1 + 100 / -x
    return x if x > 1 else None


def american(decimal):
    """Momio decimal → americano: 2.50 → +150, 1.80 → -125."""
    return (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)


def odds_text(decimal):
    """Momio para mostrar en el campo de texto: americano si representa el valor exacto, si no decimal."""
    text = f"{american(decimal):+.0f}"
    return text if abs(decimal_odds(text) - decimal) < 1e-6 else f"{decimal:g}"


def kelly(p, decimal):
    """Fracción del bank a apostar con probabilidad de ganar p: (p × momio − 1) / (momio − 1).

    Es 0 cuando el momio paga menos de lo que vale la probabilidad (no hay ventaja).
    """
    return max((p * decimal - 1) / (decimal - 1), 0)


def load_odds():
    if not ODDS_PATH.exists():
        return pd.DataFrame(columns=ODDS_COLS)
    return pd.read_csv(ODDS_PATH, dtype={"GAME_ID": str})


def save_odds(game_id, game_date, team, odds, model_p):
    """Guarda el momio de un equipo y la probabilidad que le daba el modelo; reemplaza el anterior de ese equipo."""
    log = load_odds()
    log = log[~((log["GAME_ID"] == game_id) & (log["TEAM"] == team))]
    row = pd.DataFrame([[game_id, game_date, team, odds, round(model_p, 4), dt.datetime.now().isoformat(timespec="seconds")]],
                       columns=ODDS_COLS)
    pd.concat([log, row] if len(log) else [row], ignore_index=True).to_csv(ODDS_PATH, index=False)


def odds_results(odds, games):
    """Momios guardados de partidos ya terminados, con la probabilidad del mercado y el resultado.

    La probabilidad del mercado es 1 / momio; si están los momios de los dos equipos se le quita el
    margen de la casa repartiéndolo para que sumen 100 %. PROFIT es lo que habría dejado apostar 1
    unidad cuando el modelo veía ventaja (su probabilidad mayor que la que implica el momio).
    """
    df = odds.merge(games[["GAME_ID", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN"]], on="GAME_ID")
    df["WON"] = np.where(df["HOME_WIN"] == 1, df["HOME_TEAM"], df["AWAY_TEAM"]) == df["TEAM"]
    implied = 1 / df["ODDS"]
    df["BOTH"] = df.groupby("GAME_ID")["TEAM"].transform("size") == 2
    df["MARKET_P"] = np.where(df["BOTH"], implied / implied.groupby(df["GAME_ID"]).transform("sum"), implied)
    df["EDGE"] = df["MODEL_P"] - implied
    df["PROFIT"] = np.where(df["EDGE"] > 0, np.where(df["WON"], df["ODDS"] - 1, -1.0), np.nan)
    return df


def market_summary(results):
    """Modelo contra mercado en los partidos con momio de los dos equipos, y las apuestas con ventaja."""
    both = results[results["BOTH"]]
    one_side = both.drop_duplicates("GAME_ID")  # con un lado basta: el otro es el complemento
    bets = results["PROFIT"].dropna()
    return {
        "Partidos con los dos momios": len(one_side),
        "Aciertos modelo": ((one_side["MODEL_P"] >= 0.5) == one_side["WON"]).mean(),
        "Aciertos mercado": ((one_side["MARKET_P"] >= 0.5) == one_side["WON"]).mean(),
        "Brier modelo": ((one_side["MODEL_P"] - one_side["WON"]) ** 2).mean(),
        "Brier mercado": ((one_side["MARKET_P"] - one_side["WON"]) ** 2).mean(),
        "Apuestas con ventaja": len(bets),
        "Ganancia por unidad": bets.mean() if len(bets) else np.nan,
    }
