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


def set_heatmap_annotation_colors(ax, threshold=0.5):
    collection = ax.collections[0]
    norm = collection.norm
    cmap = collection.cmap
    for text in ax.texts:
        try:
            value = float(text.get_text())
        except ValueError:
            continue
        r, g, b, _ = cmap(norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text.set_color("white" if luminance < threshold else "#222222")
        text.set_fontsize(ANNOTATION_SIZE)
        text.set_fontweight("normal")


apply_paper_style()

# ---------------------------------- figure ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "kernel_size_dataset_allscc"

# Inlined from kernel_size_dataset_allscc.csv
data = {
    "kernel_size": [3, 5, 7],
    "GC": [0.132, 0.236, 0.188],
    "ESRD": [0.192, 0.256, 0.208],
    "CRC": [0.182, 0.241, 0.194],
    "IBD": [0.358, 0.379, 0.347],
}


def main():
    df = pd.DataFrame(data).set_index("kernel_size")
    sns.set_theme(style="white", font="Times New Roman")

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=300)
    heatmap = sns.heatmap(
        df,
        ax=ax,
        cmap="viridis",
        vmin=0.1,
        vmax=0.4,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 18, "fontweight": "bold"},
        linewidths=1.6,
        linecolor="white",
        cbar_kws={
            "label": "ALL_SCC",
            "ticks": [0.1, 0.2, 0.3, 0.4],
            "shrink": 0.94,
            "pad": 0.06,
        },
    )

    ax.set_xlabel("Dataset", fontsize=21, fontweight="bold", labelpad=14)
    ax.set_ylabel("Kernel size", fontsize=21, fontweight="bold", labelpad=16)
    ax.set_xticklabels(df.columns, fontsize=18, rotation=0)
    ax.set_yticklabels(df.index.astype(str), fontsize=18, rotation=0)
    ax.tick_params(axis="both", length=0)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16, length=0, pad=6)
    cbar.set_label("ALL_SCC", fontsize=19, fontweight="bold", labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.14, right=0.87, top=0.92, bottom=0.2)
    png_path, pdf_path = save_figure(fig, OUTPUT_BASE)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
