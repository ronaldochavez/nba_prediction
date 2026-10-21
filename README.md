# Predicción NBA por alineación

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)

Predice el porcentaje de victoria y el marcador de un partido de la temporada **2026-27** según los 5 jugadores que elijas.
Usa una regresión logística entrenada con las stats individuales de temporadas anteriores y una proyección de puntos basada en efectividad ofensiva, defensiva y posesiones.

![Pestaña Predicción](docs/prediccion.png)

## Qué hace

- **Jugadores**: compara las stats por partido de un jugador entre 2024-25 y 2025-26, con la diferencia entre ambas y una gráfica.
- **Predicción**: eliges una fecha y un partido del calendario 2026-27, armas la alineación de cada equipo y ves el porcentaje de victoria,
  los puntos proyectados de cada equipo y el total del partido. Por defecto se sugieren los 5 jugadores con más minutos en 2025-26.
  Debajo se muestra qué stats inclinan la balanza hacia cada equipo.

![Pestaña Jugadores](docs/jugadores.png)

## Resultados

Cada temporada se predice con un modelo entrenado solo con las temporadas anteriores:

| Temporada de prueba | Entrenamiento | Modelo | Siempre gana el local |
|---|---|---|---|
| 2023-24 | 2021-22 a 2022-23 | 66.5 % | 54.3 % |
| 2024-25 | 2021-22 a 2023-24 | 66.7 % | 54.4 % |
| 2025-26 | 2021-22 a 2024-25 | 66.3 % | 55.4 % |
| **Promedio** | | **66.5 %** | 54.7 % |

El modelo final, el que usa la app para 2026-27, se entrena con los 6,150 partidos de 2021-22 a 2025-26.

## Cómo funciona

1. **Alineación**: en cada partido histórico se toman los 5 jugadores con más minutos de cada equipo.
2. **Features**: la diferencia local − visitante del promedio de PTS, REB, AST, STL, BLK, TOV, MIN, FG%, 3P% y +/- de esos 5 jugadores.
3. **Stats de la temporada anterior**: cada partido se describe con las stats de la temporada previa (los de 2025-26 con las de 2024-25, los de 2026-27 con las de 2025-26, etc.).
   Así el modelo se evalúa igual que como se usa: los promedios de la misma temporada ya incluyen el resultado del partido que se predice e inflan el acierto de forma artificial.
4. **Modelo**: regresión logística sobre las diferencias estandarizadas, con regularización elegida por validación cruzada.
   Las stats están muy correlacionadas (PTS y MIN: 0.90) y, sin regularización, el modelo repartía el peso entre ellas al azar.

   ```
   z = ventaja de local + Σ peso × diferencia estandarizada
   P(gana el local) = 1 / (1 + e^(−z))
   ```

5. **Jugadores sin stats** (novatos o quienes no jugaron la temporada anterior) cuentan como un jugador de banca promedio (menos de 15 minutos por partido).

### Cuánto pesa cada stat

| Stat | Peso | Importancia |
|---|---|---|
| +/- | +0.27 | 25 % |
| FG% | +0.17 | 16 % |
| Puntos | +0.17 | 16 % |
| Robos | +0.16 | 15 % |
| Asistencias | +0.09 | 8 % |
| Tapones | +0.06 | 6 % |
| Rebotes | +0.06 | 6 % |
| Minutos | +0.04 | 3 % |
| Pérdidas | +0.03 | 3 % |
| 3P% | +0.02 | 2 % |

El peso es el cambio en `z` por cada desviación estándar de diferencia entre las alineaciones; la ventaja de local vale +0.24.
Que las pérdidas sumen a favor no es un error: las cometen más los jugadores que manejan mucho el balón, que suelen ser las estrellas.

## Proyección de puntos

Usa la efectividad ofensiva (puntos anotados por 100 posesiones), la defensiva (puntos recibidos por 100 posesiones) y el ritmo (posesiones por partido):

```
posesiones      = promedio del ritmo de ambos equipos
puntos de A     = posesiones × (ofensiva de A + defensiva de B) / 2 / 100  ± ventaja de local / 2
total           = puntos de A + puntos de B
```

Se calcula de dos formas, siempre con los ratings de la temporada anterior:

- **Por equipo**: ratings del equipo completo. No refleja los traspasos del verano.
- **Por alineación**: promedio de los ratings en cancha de los 5 jugadores elegidos. Los jugadores con pocas posesiones se acercan al promedio de la liga, porque su rating es muy ruidoso.

Error medio contra el marcador real (2023-24 a 2025-26, cada temporada con los ratings de la anterior):

| Método | Puntos por equipo | Total del partido | Sesgo del total |
|---|---|---|---|
| Promedio de la liga (referencia) | 10.3 | 16.0 | −2.0 |
| Por equipo | 10.0 | 15.7 | −2.0 |
| Por alineación | 10.0 | 15.7 | −1.7 |

El marcador de un partido de la NBA varía mucho: incluso con los ratings reales de la misma temporada, que no se conocen de antemano,
el error sería de 9.3 puntos por equipo y 14.8 en el total. El sesgo negativo viene de que la liga anota un poco más cada temporada.
Para elegir al ganador conviene el porcentaje de victoria (66.5 % de aciertos), no la diferencia del marcador proyectado (60–61 %).

## Instalación y uso

El repositorio ya incluye los datos (descargados en octubre de 2026) y el modelo entrenado.

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. *(Opcional)* Descarga datos nuevos, reentrena el modelo y evalúa la proyección de puntos:
   ```bash
   python data.py
   python model.py
   python points.py
   ```
3. Abre la app:
   ```bash
   streamlit run app.py
   ```

`data.py` no vuelve a descargar los archivos que ya existen. Para actualizar uno (por ejemplo, las plantillas después de los cortes de pretemporada), bórralo de `data/` y vuelve a ejecutar `python data.py`.

## Estructura

```
├── data.py             # Descarga los datos de nba_api y los guarda en data/
├── model.py            # Entrena y evalúa el modelo de victoria → model.joblib
├── points.py           # Proyección de puntos y su evaluación
├── app.py              # App de Streamlit
├── roster_moves.csv    # Movimientos de plantilla que la API aún no refleja
├── data/               # CSV descargados
└── docs/               # Capturas para este README
```

## Datos

Los datos vienen de [stats.nba.com](https://www.nba.com/stats) a través de [nba_api](https://github.com/swar/nba_api), con pausas entre llamadas para respetar los límites:

| Archivo | Contenido |
|---|---|
| `player_stats_<temporada>.csv` | Stats por partido de cada jugador (2020-21 a 2025-26) |
| `player_ratings_<temporada>.csv` | Ratings ofensivo, defensivo y ritmo del equipo con cada jugador en cancha (2020-21 a 2025-26) |
| `team_ratings_<temporada>.csv` | Ratings ofensivo, defensivo y ritmo de cada equipo (2020-21 a 2025-26) |
| `player_games_<temporada>.csv` | Qué jugadores participaron en cada partido y cuántos minutos (2021-22 a 2025-26) |
| `games_<temporada>.csv` | Local, visitante y marcador de cada partido (2021-22 a 2025-26) |
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

## Licencia

[MIT](LICENSE) © 2026 Ronaldo Chavez

---

Proyecto personal sin relación con la NBA.
