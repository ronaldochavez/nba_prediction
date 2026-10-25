"""Momios y tamaño de apuesta con el criterio de Kelly."""

# Fracción de Kelly: la completa maximiza el crecimiento del bank solo si las probabilidades son exactas
KELLY_FRACTIONS = {"Completo": 1, "1/2": 0.5, "1/4": 0.25}


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


def kelly(p, decimal):
    """Fracción del bank a apostar con probabilidad de ganar p: (p × momio − 1) / (momio − 1).

    Es 0 cuando el momio paga menos de lo que vale la probabilidad (no hay ventaja).
    """
    return max((p * decimal - 1) / (decimal - 1), 0)
