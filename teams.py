"""Logo y color de cada equipo de la NBA y foto de cada jugador para la app."""

# Abreviatura: (id de la NBA, color del equipo lo bastante oscuro para leer texto blanco encima)
TEAMS = {
    "ATL": (1610612737, "#C8102E"), "BOS": (1610612738, "#007A33"), "BKN": (1610612751, "#111111"),
    "CHA": (1610612766, "#1D1160"), "CHI": (1610612741, "#CE1141"), "CLE": (1610612739, "#860038"),
    "DAL": (1610612742, "#00538C"), "DEN": (1610612743, "#0E2240"), "DET": (1610612765, "#1D42BA"),
    "GSW": (1610612744, "#1D428A"), "HOU": (1610612745, "#CE1141"), "IND": (1610612754, "#002D62"),
    "LAC": (1610612746, "#C8102E"), "LAL": (1610612747, "#552583"), "MEM": (1610612763, "#12173F"),
    "MIA": (1610612748, "#98002E"), "MIL": (1610612749, "#00471B"), "MIN": (1610612750, "#0C2340"),
    "NOP": (1610612740, "#0C2340"), "NYK": (1610612752, "#006BB6"), "OKC": (1610612760, "#007AC1"),
    "ORL": (1610612753, "#0077C0"), "PHI": (1610612755, "#006BB6"), "PHX": (1610612756, "#1D1160"),
    "POR": (1610612757, "#C8102E"), "SAC": (1610612758, "#5A2D81"), "SAS": (1610612759, "#3A3D40"),
    "TOR": (1610612761, "#CE1141"), "UTA": (1610612762, "#002B5C"), "WAS": (1610612764, "#002B5C"),
}


def logo_url(abbr, dark=False):
    """Logo oficial del CDN de la NBA. Lo pide el navegador, así que también funciona en Streamlit Cloud.

    La NBA tiene una versión para fondo oscuro: el logo de Brooklyn, por ejemplo, es negro en la normal.
    """
    return f"https://cdn.nba.com/logos/nba/{TEAMS[abbr][0]}/primary/{'D' if dark else 'L'}/logo.svg"


def headshot_url(player_id):
    """Foto oficial del jugador en el CDN de la NBA; a quien no tiene foto le da una silueta."""
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
