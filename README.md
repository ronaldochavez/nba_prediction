# Predicción NBA por alineación

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)

Predice el porcentaje de victoria y el marcador de un partido de la temporada **2026-27** según los 5 jugadores que elijas.
Usa una regresión logística entrenada con las stats individuales de los jugadores y una proyección de puntos basada en efectividad ofensiva, defensiva y posesiones.
Ambas mezclan la temporada pasada con lo que va de la actual, y la app lleva el seguimiento de sus aciertos en la temporada real.

![Pestaña Predicción](docs/prediccion.png)

## Qué hace

- **Jugadores**: compara las stats por partido de un jugador entre 2024-25 y 2025-26, y muestra su **forma reciente**:
  promedio de los últimos 5 y 10 partidos (pretemporada, temporada regular y playoffs), la tendencia frente a su promedio
  de la temporada pasada, una vista **por 36 minutos** y el detalle partido a partido.
- **Predicción**: eliges una fecha y un partido del calendario 2026-27, armas la alineación de cada equipo y ves el porcentaje de victoria,
  los puntos proyectados de cada equipo y el total del partido. Por defecto se sugieren los 5 jugadores con más minutos esperados.
  Debajo se muestra qué stats inclinan la balanza hacia cada equipo.
- **Seguimiento**: compara cada predicción con el resultado real de los partidos ya jugados de 2026-27.

![Pestaña Jugadores](docs/jugadores.png)

![Forma reciente](docs/forma.png)

![Seguimiento](docs/seguimiento.png)

## Resultados

Cada temporada se predice con un modelo entrenado solo con las temporadas anteriores y, en cada partido, con lo que se sabía antes de jugarlo:

| Temporada de prueba | Entrenamiento | Modelo | Siempre gana el local |
|---|---|---|---|
| 2023-24 | 2021-22 a 2022-23 | 67.2 % | 54.3 % |
| 2024-25 | 2021-22 a 2023-24 | 68.7 % | 54.4 % |
| 2025-26 | 2021-22 a 2024-25 | 70.7 % | 55.4 % |
| **Promedio** | | **68.9 %** | 54.7 % |

El modelo final, el que usa la app para 2026-27, se entrena con los 6,150 partidos de 2021-22 a 2025-26.

## Cómo funciona

1. **Alineación**: en cada partido histórico se toman los 5 jugadores con más minutos de cada equipo.
2. **Features**: la diferencia local − visitante del promedio de PTS, REB, AST, STL, BLK, TOV, MIN, FG%, 3P% y +/- de esos 5 jugadores.
3. **Temporada anterior + temporada en curso**: las stats de cada jugador mezclan su temporada anterior con lo que llevaba
   de la actual antes de ese partido. La actual pesa más con cada partido jugado: a los 10 partidos, mitad y mitad.

   | Peso de la temporada en curso | Aciertos (2023-24 a 2025-26) |
   |---|---|
   | Ninguno (solo temporada anterior) | 66.1 % |
   | **Mitad y mitad a los 10 partidos** | **68.9 %** |
   | Mitad y mitad a los 5 o 20 partidos | 68.4 % / 68.6 % |

   Nunca se usan las stats del partido que se predice: los promedios de la temporada completa ya incluyen su resultado e inflan el acierto de forma artificial.
4. **Modelo**: regresión logística sobre las diferencias estandarizadas, con regularización elegida por validación cruzada.
   Las stats están muy correlacionadas (PTS y MIN: 0.90) y, sin regularización, el modelo repartía el peso entre ellas al azar.

   ```
   z = ventaja de local + Σ peso × diferencia estandarizada
   P(gana el local) = 1 / (1 + e^(−z))
   ```

5. **Jugadores que no jugaron la temporada anterior** (por ejemplo, por una lesión) usan sus stats de la temporada previa a esa.
   Los que no tienen ninguna de las dos (novatos) parten como un jugador de banca promedio (menos de 15 minutos por partido) hasta que acumulan partidos.

### Cuánto pesa cada stat

| Stat | Peso | Importancia |
|---|---|---|
| +/- | +0.55 | 43 % |
| Puntos | +0.14 | 11 % |
| FG% | +0.12 | 10 % |
| Robos | +0.10 | 8 % |
| Rebotes | +0.09 | 7 % |
| Asistencias | +0.08 | 6 % |
| Minutos | +0.07 | 6 % |
| Tapones | +0.07 | 5 % |
| Pérdidas | −0.05 | 4 % |
| 3P% | −0.00 | 0 % |

El peso es el cambio en `z` por cada desviación estándar de diferencia entre las alineaciones; la ventaja de local vale +0.25.
El +/- de la temporada en curso es, por mucho, lo que mejor anticipa al ganador.

### Alineación sugerida

Se sugieren los 5 jugadores con más **minutos esperados**: el promedio de minutos de las dos temporadas anteriores, ponderado por
partidos jugados (con 20 partidos ficticios de 10 minutos para que una muestra corta no pese de más), mezclado con los minutos de esta
temporada, que pesan mucho desde el primer partido. Es la regla que mejor acertó a los 5 con más minutos de cada partido en 2024-25 y
2025-26 (3.97 de 5): no deja fuera a una estrella que se perdió media temporada por lesión ni sube a un jugador con pocos partidos.

## Proyección de puntos

Usa la efectividad ofensiva (puntos anotados por 100 posesiones), la defensiva (puntos recibidos por 100 posesiones) y el ritmo (posesiones por partido):

```
puntos de A = ((ataque A × ritmo A) / 100 + (defensa B × ritmo B) / 100) / 2  ± ventaja de local / 2
total       = puntos de A + puntos de B
```

Es decir, el promedio entre lo que A suele anotar por partido y lo que B suele recibir. Por ejemplo, MIL @ WAS con los ratings de 2025-26:
MIL = (112.2 × 98.3 / 100 + 121.5 × 102.5 / 100) / 2 = 117.4 puntos, antes de la ventaja de local.

Se calcula de dos formas:

- **Por equipo**: ratings del equipo, mezclando la temporada anterior con lo que va de la actual (calculado con los box scores, partido a partido).
- **Por alineación**: promedio de los ratings en cancha de los 5 jugadores elegidos. Los jugadores con pocas posesiones se acercan al promedio de la liga,
  porque su rating es muy ruidoso. La API solo da los ratings de la temporada en curso acumulados a hoy, así que la app los mezcla, pero la evaluación
  histórica de este método usa solo la temporada anterior.

Error medio contra el marcador real (2023-24 a 2025-26, con lo que se sabía antes de cada partido):

| Método | Puntos por equipo | Total del partido | Sesgo del total |
|---|---|---|---|
| Promedio de la liga (referencia) | 10.3 | 16.0 | −2.0 |
| **Por equipo** | **9.6** | **15.2** | −1.3 |
| Por alineación | 10.0 | 15.7 | −1.6 |

El marcador de un partido de la NBA varía mucho: incluso con los ratings reales de la temporada completa, que no se conocen de antemano,
el error sería de 9.3 puntos por equipo y 14.8 en el total. El sesgo negativo viene de que la liga anota un poco más cada temporada.
Para elegir al ganador conviene el porcentaje de victoria (68.9 % de aciertos), no la diferencia del marcador proyectado (65.9 % por equipo).

## Seguimiento de la temporada

La pestaña **Seguimiento** (y `python tracking.py`) compara cada partido ya jugado de 2026-27 con lo que la app habría predicho:
ganador, marcador por equipo y por alineación, y total. Cada predicción usa solo lo que se sabía antes del partido y la alineación
que de verdad jugó, igual que la evaluación histórica. Con pocos partidos el porcentaje de aciertos varía mucho.

## Instalación y uso

El repositorio ya incluye los datos (descargados en octubre de 2026) y el modelo entrenado.

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. *(Opcional)* Descarga datos nuevos, reentrena y evalúa:
   ```bash
   python data.py
   python model.py
   python points.py
   python tracking.py
   ```
3. Abre la app:
   ```bash
   streamlit run app.py
   ```

`data.py` vuelve a descargar siempre lo de la temporada actual (resultados, box scores, ratings, plantillas y calendario), que cambia a diario;
los archivos históricos solo se bajan una vez. Si la app está abierta, reiníciala para que lea los datos nuevos.

## Estructura

```
├── data.py             # Descarga los datos de nba_api y los guarda en data/
├── model.py            # Entrena y evalúa el modelo de victoria → model.joblib
├── points.py           # Proyección de puntos y su evaluación
├── tracking.py         # Predicciones contra resultados de la temporada actual
├── app.py              # App de Streamlit
├── roster_moves.csv    # Movimientos de plantilla que la API aún no refleja
├── data/               # CSV descargados
└── docs/               # Capturas para este README
```

## Datos

Los datos vienen de [stats.nba.com](https://www.nba.com/stats) a través de [nba_api](https://github.com/swar/nba_api), con pausas entre llamadas para respetar los límites:

| Archivo | Contenido |
|---|---|
| `player_stats_<temporada>.csv` | Stats por partido de cada jugador (2019-20 a 2025-26) |
| `player_ratings_<temporada>.csv` | Ratings ofensivo, defensivo y ritmo del equipo con cada jugador en cancha (2019-20 a 2026-27) |
| `team_ratings_<temporada>.csv` | Ratings ofensivo, defensivo y ritmo de cada equipo (2019-20 a 2025-26) |
| `game_logs_<temporada>.csv` | Box score de cada jugador en cada partido, con pretemporada y playoffs (2021-22 a 2026-27) |
| `games_<temporada>.csv` | Local, visitante y marcador de cada partido jugado (2021-22 a 2026-27) |
| `schedule_2026-27.csv` | Calendario de la temporada actual |
| `rosters_2026-27.csv` | Plantillas actuales |

Si hay fichajes, traspasos o cortes que la API todavía no refleja, agrégalos a `roster_moves.csv`:
- `FROM_TEAM`: equipo donde la API tiene hoy al jugador (vacío si no está en ninguna plantilla).
- `TO_TEAM`: equipo nuevo (vacío si queda libre).
- `PLAYER_ID`: el número que aparece en la URL del jugador en nba.com.

Un movimiento deja de aplicarse solo cuando la API ya lo refleja.

## Limitaciones y próximos pasos

- No considera la química entre jugadores (planeado para la versión 2).
- No tiene en cuenta lesiones, descansos ni partidos consecutivos.
- Las stats de los 5 jugadores se promedian sin ponderar por minutos.
- Próximo: "efectividad de minutos" (puntos por minuto) por jugador, por alineación y por equipo completo.

## Licencia

[MIT](LICENSE) © 2026 Ronaldo Chavez

---

Proyecto personal sin relación con la NBA.
