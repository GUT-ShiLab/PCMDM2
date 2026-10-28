from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PNG_PATH = BASE_DIR / "ALL_SCC_cross_dataset_bar.png"
PDF_PATH = BASE_DIR / "ALL_SCC_cross_dataset_bar.pdf"

data = {
    "Dataset": ["CRC→GC", "GC→CRC", "CRC→IBD", "IBD→CRC", "GC→IBD", "IBD→GC"],
    "MelonnPan": [0.2654, 0.2464, 0.2919, 0.1671, 0.3112, 0.1636],
    "ENVIM": [0.3235, 0.2768, 0.2647, 0.1788, 0.2227, 0.1808],
    "MMINP": [0.3720, 0.2722, 0.3852, 0.1628, 0.3641, 0.1625],
    "MiMeNet": [0.1830, 0.1192, 0.1738, 0.1236, 0.1533, 0.1068],
    "BiomeNED": [0.1136, 0.1514, 0.0664, 0.1001, 0.1154, 0.0576],
    "mNODE": [0.1013, 0.1000, 0.2383, 0.1425, 0.1829, 0.1487],
    "LOCATE": [0.1072, 0.0736, 0.1848, 0.1324, 0.1621, 0.1226],
    "PCMDM2": [0.1807, 0.1345, 0.2302, 0.1867, 0.1865, 0.1578],
}

methods = ["MelonnPan", "ENVIM", "MMINP", "MiMeNet", "BiomeNED", "mNODE", "LOCATE", "PCMDM2"]

colors = {
    "MelonnPan": "#4C84D3",
    "ENVIM": "#F0A11C",
    "MMINP": "#D95C45",
    "MiMeNet": "#EA8B69",
    "BiomeNED": "#F04A4A",
    "mNODE": "#B71C1C",
    "LOCATE": "#8EC5FF",
    "PCMDM2": "#E60000",
}


def main():
    plt.rcParams.update({"font.family": "Times New Roman"})

    df = pd.DataFrame(data)
    x = np.arange(len(df["Dataset"]))
    bar_width = 0.09

    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=300)

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_width
        is_pcmdm2 = method == "PCMDM2"
        ax.bar(
            x + offset,
            df[method],
            width=bar_width,
            label=method,
            color=colors[method],
            edgecolor="black" if is_pcmdm2 else "white",
            linewidth=1.0 if is_pcmdm2 else 0.5,
        )

    ax.set_ylabel("ALL_SCC", fontsize=12, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 0.42)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=9,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=1.5,
        handletextpad=0.5,
        handlelength=1.4,
    )

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.98)

    fig.savefig(PNG_PATH, dpi=900, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
