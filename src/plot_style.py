"""Shared plotting style for AlphaLens charts."""

from __future__ import annotations

import matplotlib.pyplot as plt

NAVY = "#1f3a5f"
GOLD = "#c9a84c"
MIN_FONT_SIZE = 12
FIG_DPI = 300


def apply_plot_style() -> None:
    """Apply a consistent visual style across all charts."""
    plt.rcParams.update(
        {
            "font.size": MIN_FONT_SIZE,
            "axes.titlesize": MIN_FONT_SIZE + 2,
            "axes.labelsize": MIN_FONT_SIZE,
            "xtick.labelsize": MIN_FONT_SIZE,
            "ytick.labelsize": MIN_FONT_SIZE,
            "legend.fontsize": MIN_FONT_SIZE,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
        }
    )

