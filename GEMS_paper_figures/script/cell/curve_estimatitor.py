
import json
from pathlib import Path
from typing import List, Tuple
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────── Configuration ──────────────── #
np.random.seed(42)

LABEL_FONT_OPTIONS = {'weight': 'bold', 'size': 10, 'family': 'Helvetica'}
AX_X_FONTSIZE    = 10
AX_Y_FONTSIZE    = 10
LEGEND_FONT_SIZE = 10
XAXIS_LABEL_NUM  = 7

WIDTH           = 560/72                                         # figure width in inches
HEIGHT          = 225/72                                          # figure height in inches

plt.rcParams.update({
    'figure.figsize': (WIDTH, HEIGHT),
    'font.size': AX_X_FONTSIZE,
    'axes.titlesize': LABEL_FONT_OPTIONS['size'],
    'axes.labelsize': AX_Y_FONTSIZE,
    'xtick.labelsize': AX_X_FONTSIZE,
    'ytick.labelsize': AX_Y_FONTSIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'axes.titleweight': LABEL_FONT_OPTIONS['weight'],
    'axes.labelweight': LABEL_FONT_OPTIONS['weight'],
    'font.family': 'Helvetica',
})

GRAY             = "#CCCCCC"                              # non-highlight colour
HIGHLIGHT_PALETTE = list(plt.cm.tab10.colors)             # colour cycle

DATA_DIR   = Path("data/241106_CCDS_253g1-hek293a_report/growth_curve")
OUTPUT_DIR = Path("figure/cell_curve_estimator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────── #


def load_curve(json_file: Path) -> pd.DataFrame:
    """Read the CSV referenced in *json_file* and return a tidy DataFrame."""
    with open(json_file) as f:
        meta = json.load(f)
    df = pd.read_csv(meta["csv_path"])
    df["time"] = pd.to_datetime(df["time"], unit="m")
    return (
        df.sort_values("time")[["time", "density"]]
          .dropna()
          .reset_index(drop=True)
    )


def vis_density_curve(ax: plt.Axes, df: pd.DataFrame, label: str, colour: str) -> None:
    """Plot one time-series density curve on *ax*."""
    ax.plot(df["time"], df["density"], ".-", lw=2, color=colour, label=label)
    ax.set_ylim(0, 1)

    yticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(yticks)

    # メモリ線（グリッド）を描画
    ax.set_axisbelow(True)  # グリッドを曲線の背面に
    ax.yaxis.grid(True,
                  color=GRAY,
                  linestyle='--',
                  linewidth=0.5,
                  which='major')

    total_days = (df["time"].max().ceil("D") - df["time"].min().floor("D")).days
    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=max(total_days // XAXIS_LABEL_NUM, 1))
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
    ax.tick_params(axis="x", labelsize=AX_X_FONTSIZE, rotation=60)
    ax.tick_params(axis="y", labelsize=AX_Y_FONTSIZE)



def plot_group(
    ax: plt.Axes,
    highlight: str,
    groups: List[Tuple[str, List[Path]]],
) -> None:
    """Draw HEK / hiPSC curves on *ax*, emphasising *highlight* group."""
    colour_cycle = cycle(HIGHLIGHT_PALETTE)

    for gname, gfiles in groups:
        label_name = "hiPSC" if gname == "iPS" else "HEK293A"
        for idx, f in enumerate(gfiles):
            df = load_curve(f)
            if gname == highlight:
                colour = next(colour_cycle)         # distinct colour per curve
            else:
                colour = GRAY                       # de-emphasised
            vis_density_curve(ax, df, f"{label_name}_{idx}", colour)

    # ---- legend ---- #
    handles, labels = ax.get_legend_handles_labels()

    keep_handles = [
        h for h, l in zip(handles, labels) if highlight in l
    ]  # only highlight curves

    other = "iPS" if highlight == "HEK" else "HEK"
    other_labels = "hiPSC" if other == "iPS" else "HEK293A"
    if any(other in l for l in labels):
        proxy = plt.Line2D([], [], color=GRAY, lw=2, label=f"{other_labels}")
        keep_handles.append(proxy)

    ax.legend(handles=keep_handles, fontsize=LEGEND_FONT_SIZE, loc="upper right")
    ax.set_title(f"{highlight} emphasised", **LABEL_FONT_OPTIONS)
    ax.set_ylabel("Density", **LABEL_FONT_OPTIONS)



def main() -> None:
    # ---- gather JSON files ------------------------------------------------ #
    all_json = list(DATA_DIR.glob("*.json"))
    hek_files = sorted([p for p in all_json if "HEK" in p.stem])
    ips_files = sorted([p for p in all_json if "iPS" in p.stem])
    groups = [("HEK", hek_files), ("iPS", ips_files)]

    # ---- two-panel figure -------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)

    plot_group(axes[0], "HEK", groups)   # left  : HEK highlighted
    plot_group(axes[1], "iPS", groups)   # right : iPS highlighted

    # Right panel also shows Y-axis ticks/labels (sharey=True keeps limits)
    axes[1].tick_params(axis="y", labelsize=AX_Y_FONTSIZE)
    axes[1].set_ylabel("")  # clear Y-axis label for right panel
    axes[0].set_title("HEK293A", **LABEL_FONT_OPTIONS)
    axes[1].set_title("hiPSC", **LABEL_FONT_OPTIONS)

    # Add centered x-axis label
    fig.text(0.53, 0.01, "Date", ha='center', **LABEL_FONT_OPTIONS)

    plt.tight_layout()
    # fig.savefig(OUTPUT_DIR / "density_dual_coloured.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "Fig5-E.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()