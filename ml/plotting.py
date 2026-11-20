"""Shared, consistent styling for every saved figure."""

import matplotlib

matplotlib.use("Agg")  # headless backend: works in scripts, CI and on servers

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed colour roles (colour-blind-validated categorical slots 1 and 2).
NO_DEFAULT_COLOR = "#2a78d6"  # blue
DEFAULT_COLOR = "#eb6834"  # orange
CLASS_PALETTE = {0: NO_DEFAULT_COLOR, 1: DEFAULT_COLOR}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled
NEUTRAL = "#52514e"

# Diverging map for correlations: blue (-1) -> neutral grey (0) -> orange (+1).
DIVERGING = LinearSegmentedColormap.from_list("credence_div", ["#2a78d6", "#f0efec", "#eb6834"])
# Sequential single-hue map for counts (confusion matrices).
SEQUENTIAL = LinearSegmentedColormap.from_list("credence_seq", ["#f4f8fd", "#2a78d6", "#0d3a73"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "xtick.color": NEUTRAL,
            "ytick.color": NEUTRAL,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
