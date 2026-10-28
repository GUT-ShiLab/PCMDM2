from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------- paper plot style -----------------------------
FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans"]
TITLE_SIZE = 10
AXIS_LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 7.5

GRID_COLOR = "#D8D8D8"
TEXT_COLOR = "#222222"


def _available_font_family():
    available = {f.name for f in fm.fontManager.ttflist}
    for font in FONT_FAMILY:
        if font in available:
            return font
    return "DejaVu Sans"


def apply_paper_style():
    font = _available_font_family()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Liberation Sans", "DejaVu Sans"],
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "normal",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.6,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )
    return font


def save_figure(fig, output_base, vector="pdf"):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    vector_path = output_base.with_suffix(f".{vector}")
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.05, "facecolor": "white"}
    fig.savefig(png_path, dpi=600, **kwargs)
    fig.savefig(vector_path, dpi=600, **kwargs)
    return png_path, vector_path


apply_paper_style()

# ---------------------------------- figure ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "traversal_strategy_dataset_allscc"

# Inlined from traversal_strategy_dataset_allscc.csv
data = {
    "traversal_strategy": [
        "postorder",
        "levelorder",
        "ladderization",
        "postorder+levelorder",
        "postorder+ladderization",
        "ladderization+levelorder",
        "postorder+ladderization+levelorder",
    ],
    "IBD": [0.3794, 0.3397, 0.3398, 0.3503, 0.3488, 0.3589, 0.3464],
    "GC": [0.2364, 0.2089, 0.1814, 0.2077, 0.1351, 0.2251, 0.1964],
    "ESRD": [0.2556, 0.1959, 0.1839, 0.1920, 0.2203, 0.2032, 0.2044],
    "CRC": [0.2413, 0.2106, 0.1866, 0.1856, 0.1864, 0.1835, 0.1857],
}


def main():
    df = pd.DataFrame(data).set_index("traversal_strategy")
    sns.set_theme(style="white", font="Times New Roman")

    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=300)
    heatmap = sns.heatmap(
        df,
        ax=ax,
        cmap="viridis",
        vmin=0.1,
        vmax=0.4,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 15, "fontweight": "bold"},
        linewidths=1.5,
        linecolor="white",
        cbar_kws={
            "label": "ALL_SCC",
            "ticks": [0.1, 0.2, 0.3, 0.4],
            "shrink": 0.94,
            "pad": 0.055,
        },
    )

    ax.set_xlabel("Dataset", fontsize=21, fontweight="bold", labelpad=14)
    ax.set_ylabel("Traversal strategy", fontsize=21, fontweight="bold", labelpad=18)
    ax.set_xticklabels(df.columns, fontsize=18, rotation=0)
    ax.set_yticklabels(df.index, fontsize=17, rotation=0)
    ax.tick_params(axis="both", length=0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16, length=0, pad=6)
    cbar.set_label("ALL_SCC", fontsize=19, fontweight="bold", labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.32, right=0.88, top=0.93, bottom=0.18)
    png_path, pdf_path = save_figure(fig, OUTPUT_BASE)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
