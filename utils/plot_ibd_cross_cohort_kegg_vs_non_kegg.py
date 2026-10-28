from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------- figure style -------------------------------
BAR_COLOR = "#4C8DC0"
LINE_COLOR = "#ED7D31"
GRID_COLOR = "#CCCCCC"
TEXT_COLOR = "#3A3A3A"

TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 7.5
ANNOTATION_SIZE = 7

BAR_WIDTH = 0.6
BAR_Y_HEADROOM = 1.32
LINE_Y_HEADROOM = 1.14

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "ibd_cross_cohort_kegg_vs_non_kegg"


def plot_panel(ax1, methods, all_scc, wp_nums, panel_title):
    x = np.arange(len(methods))
    bar_top = max(all_scc) * BAR_Y_HEADROOM
    line_top = max(wp_nums) * LINE_Y_HEADROOM

    ax1.bar(
        x,
        all_scc,
        color=BAR_COLOR,
        width=BAR_WIDTH,
        edgecolor="none",
        zorder=2,
        label="ALL_SCC",
    )
    ax1.set_ylabel("ALL_SCC", fontsize=LABEL_SIZE)
    ax1.set_ylim(0, bar_top)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, fontsize=TICK_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title(panel_title, fontsize=TITLE_SIZE, fontweight="normal", pad=6)
    ax1.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        wp_nums,
        color=LINE_COLOR,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        zorder=2,
        label="WP_Nums",
    )
    ax2.set_ylabel("WP_Nums", fontsize=LABEL_SIZE)
    ax2.set_ylim(0, line_top)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.grid(False)
    ax2.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=True)


    for xi, value in zip(x, all_scc):
        ax2.text(
            xi,
            (value + max(all_scc) * 0.02) / bar_top * line_top,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=TEXT_COLOR,
            zorder=5,
        )

    for xi, value in zip(x, wp_nums):
        ax2.text(
            xi,
            value + max(wp_nums) * 0.03,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=TEXT_COLOR,
            zorder=5,
        )


def save_figure(fig, output_base, vector="pdf"):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    vector_path = output_base.with_suffix(f".{vector}")
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.05, "facecolor": "white"}
    fig.savefig(png_path, dpi=600, **kwargs)
    fig.savefig(vector_path, dpi=600, **kwargs)
    return png_path, vector_path


def main():
    methods = ["MelonnPan", "ENVIM", "MMINP", "MiMeNet", "BiomeNED", "mNODE", "LOCATE", "PCMDM2"]

    ibd_kegg_wp_nums = [128, 58, 116, 122, 57, 102, 82, 132]
    ibd_kegg_all_scc = [0.3638, 0.2916, 0.3231, 0.3032, 0.1378, 0.2608, 0.2030, 0.3215]

    ibd_no_kegg_wp_nums = [529, 224, 630, 3323, 1197, 2839, 2401, 4107]
    ibd_no_kegg_all_scc = [0.3412, 0.1809, 0.3605, 0.2484, 0.1043, 0.1941, 0.1807, 0.2859]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.13, top=0.94, wspace=0.18)
    plot_panel(
        axes[0],
        methods,
        ibd_kegg_all_scc,
        ibd_kegg_wp_nums,
        "(a)IBD Cross-study validation ( Metabolites with KEGG annotations)",
    )
    plot_panel(
        axes[1],
        methods,
        ibd_no_kegg_all_scc,
        ibd_no_kegg_wp_nums,
        "(b)IBD Cross-study validation (Metabolites without KEGG annotations)",
    )

    png_path, pdf_path = save_figure(fig, OUTPUT_BASE)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
