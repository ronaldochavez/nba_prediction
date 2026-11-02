"""Mapas de tiros en media cancha, con las coordenadas de la NBA: décimas de pie con el aro en (0, 0).

El aro va arriba y los lados quedan como los ve el que tira: X positivo es su derecha.
"""
import base64
import io

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image

BASELINE, HALF_COURT, SIDELINE = -47.5, 422.5, 250
WIDTH, HEIGHT = 500, 470  # misma escala en los dos ejes: la cancha no se deforma
LINE_COLOR = "#8A94A6"  # se lee con el tema claro y con el oscuro
CELL = 5  # lado de cada píxel del mapa de calor: medio pie
SPREAD = 1.5  # pies que se reparte cada tiro: con pocos tiros, una mancha en la zona y no puntos sueltos
VISIBLE = 0.15  # debajo de esta intensidad no se dibuja, para que no se pinte toda la cancha
RESULTS = ["Encestó", "Falló"]
X =alt.X("X:Q", axis=None, scale=alt.Scale(domain=[-SIDELINE, SIDELINE]))
Y = alt.Y("Y:Q", axis=None, scale=alt.Scale(domain=[HALF_COURT, BASELINE]))


def arc(x, y, radius, start=0, end=360):
    """Puntos de un arco de círculo; los ángulos van en grados."""
    angles = np.radians(np.linspace(start, end, 61))
    return list(zip(x + radius * np.cos(angles), y + radius * np.sin(angles)))


def court():
    """Líneas de la media cancha."""
    three = np.degrees(np.arcsin(89.5 / 237.5))  # la línea de tres deja de ser recta a 8.95 pies del aro
    lines = [
        [(-SIDELINE, BASELINE), (SIDELINE, BASELINE), (SIDELINE, HALF_COURT), (-SIDELINE, HALF_COURT), (-SIDELINE, BASELINE)],
        [(-80, BASELINE), (-80, 142.5), (80, 142.5), (80, BASELINE)],  # pintura
        arc(0, 142.5, 60),  # círculo de tiros libres
        arc(0, 0, 40, 0, 180),  # zona restringida
        arc(0, 0, 7.5),  # aro
        [(-30, -7.5), (30, -7.5)],  # tablero
        [(-220, BASELINE), *arc(0, 0, 237.5, 180 - three, three), (220, BASELINE)],  # línea de tres
        arc(0, HALF_COURT, 60, 180, 360),  # círculo central
    ]
    points = pd.DataFrame(
        [(i, order, x, y) for i, line in enumerate(lines) for order, (x, y) in enumerate(line)],
        columns=["LINE", "ORDER", "X", "Y"],
    )
    return alt.Chart(points).mark_line(color=LINE_COLOR, strokeWidth=1).encode(
        x=X, y=Y, detail="LINE:N", order="ORDER:Q", tooltip=alt.value(None),
    )


def on_court(shots):
    """Los tiros desde más allá de media cancha no caben en el mapa."""
    return shots[shots["LOC_Y"] <= HALF_COURT]


def draw(layer):
    return alt.layer(layer, court()).properties(width=WIDTH, height=HEIGHT).configure_view(strokeWidth=0)


def heat_map(shots, dark=False):
    """Mapa de calor: entre más intenso, más tiros salieron de ahí. Va con raíz cuadrada para que los tiros
    pegados al aro, que son muchos más, no borren al resto de las zonas."""
    x_edges = np.arange(-SIDELINE, SIDELINE + CELL, CELL)
    y_edges = np.arange(BASELINE, HALF_COURT + CELL, CELL)
    counts = np.histogram2d(shots["LOC_X"], shots["LOC_Y"], [x_edges, y_edges])[0]
    reach = int(3 * SPREAD / CELL * 10)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) * CELL / 10 / SPREAD) ** 2)
    for axis in (0, 1):
        counts = np.apply_along_axis(np.convolve, axis, counts, kernel, mode="same")
    heat = np.sqrt(counts / counts.max()).T  # filas: de la línea de fondo a media cancha, como va de arriba abajo en el mapa
    # Va como imagen, que el navegador suaviza al agrandarla, y no como celdas: se notarían los cuadros y las rayas entre ellos
    colors = [(124, 45, 18), (234, 88, 12), (251, 146, 60), (253, 230, 138)] if dark else [
        (254, 215, 170), (251, 146, 60), (220, 38, 38), (127, 29, 29),
    ]
    channels = [np.interp(heat, [VISIBLE, 0.45, 0.7, 1], channel) for channel in zip(*colors)]
    channels.append(np.interp(heat, [VISIBLE, 0.55], [0, 255]))  # lo poco intenso se desvanece en el fondo de la app
    image = io.BytesIO()
    Image.fromarray(np.dstack(channels).astype("uint8")).save(image, "PNG")
    url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    return draw(alt.Chart(pd.DataFrame({"X": [-SIDELINE], "Y": [BASELINE], "URL": [url]})).mark_image(
        width=WIDTH, height=HEIGHT, align="left", baseline="top", aspect=False,
    ).encode(x=X, y=Y, url="URL:N", tooltip=alt.value(None)))


def dot_map(shots, made_color, missed_color):
    """Cada tiro en su lugar: círculo si entró, cruz si no."""
    shots = on_court(shots)
    period = shots["PERIOD"].map(lambda p: f"Cuarto {p}" if p <= 4 else f"Tiempo extra {p - 4}")
    clock = shots["MINUTES_REMAINING"].map("{:02d}".format) + ":" + shots["SECONDS_REMAINING"].map("{:02d}".format)
    points = pd.DataFrame({
        "X": shots["LOC_X"],
        "Y": shots["LOC_Y"],
        "Resultado": shots["SHOT_MADE_FLAG"].map({1: RESULTS[0], 0: RESULTS[1]}),
        "Tiro": shots["SHOT_PTS"].astype(str) + " puntos",
        "Distancia": (np.hypot(shots["LOC_X"], shots["LOC_Y"]) // 10).astype(int).astype(str) + " pies",
        "Momento": period + ", quedaban " + clock,
        "Partido": shots["GAME_LABEL"],
    })
    return draw(alt.Chart(points).mark_point(filled=False, strokeWidth=2, size=70 if len(points) <= 60 else 25).encode(
        x=X, y=Y,
        color=alt.Color("Resultado:N", scale=alt.Scale(domain=RESULTS, range=[made_color, missed_color]),
                        legend=alt.Legend(title=None, orient="bottom")),
        shape=alt.Shape("Resultado:N", scale=alt.Scale(domain=RESULTS, range=["circle", "M-1,-1L1,1M1,-1L-1,1"])),
        tooltip=["Partido", "Resultado", "Tiro", "Distancia", "Momento"],
    ))


def shot_summary(shots, points):
    """Encestados e intentos, como en un box score, y de dónde salieron los `points` del box score. Los tiros libres
    no tienen lugar en la cancha: son lo que falta, salvo si el box score va atrasado frente a los tiros y no alcanza."""
    made = shots["SHOT_MADE_FLAG"].sum()
    threes = shots[shots["SHOT_PTS"] == 3]
    made_threes = threes["SHOT_MADE_FLAG"].sum()
    text = f"{made}-{len(shots)} FG ({made / len(shots):.0%})"
    if len(threes):
        text += f" · {made_threes}-{len(threes)} triples ({made_threes / len(threes):.0%})"
    far = len(shots) - len(on_court(shots))
    if far:
        text += f" · {far} desde más allá de media cancha, fuera del mapa"
    from_twos, from_threes = 2 * (made - made_threes), 3 * made_threes
    if points >= from_twos + from_threes:
        free_throws = points - from_twos - from_threes
        text += f"  \n{points} pts: {from_twos} de 2 + {from_threes} de 3 + {free_throws} de tiros libres, que no salen en el mapa"
    return text
