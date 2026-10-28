

datasets = {
    "IBD": {
        "Methods": ["CNN_C", "CNN_F", "CNN_P", "CNN_O", "PCMDM2_C", "PCMDM2_F", "PCMDM2_P", "PCMDM2_O", "MLP_O"],
        "ALL_Scc": [0.3739, 0.3665, 0.3568, 0.3482, 0.3794, 0.3733, 0.3748, 0.3737, 0.3319],
        "WP_Nums": [5711, 5572, 5385, 5538, 5747, 5640, 5672, 5674, 5197]
    },
    "esrd": {
        "Methods": ["CNN_C", "CNN_F", "CNN_P", "CNN_O", "PCMDM2_C", "PCMDM2_F", "PCMDM2_P", "PCMDM2_O", "MLP_O"],
        "ALL_Scc": [0.2478, 0.2373, 0.2257, 0.2179, 0.2556, 0.2496, 0.2597, 0.2572, 0.2141],
        "WP_Nums": [124, 109, 92, 86, 114, 108, 118, 111, 86]
    },
    "crc": {
        "Methods": ["CNN_C", "CNN_F", "CNN_P", "CNN_O", "PCMDM2_C", "PCMDM2_F", "PCMDM2_P", "PCMDM2_O", "MLP_O"],
        "ALL_Scc": [0.2399, 0.2035, 0.2159, 0.1641, 0.2413, 0.2184, 0.2282, 0.1842, 0.1223],
        "WP_Nums": [115, 94, 97, 67, 116, 93, 98, 66, 45]
    },
    "gc": {
        "Methods": ["CNN_C", "CNN_F", "CNN_P", "CNN_O", "PCMDM2_C", "PCMDM2_F", "PCMDM2_P", "PCMDM2_O", "MLP_O"],
        "ALL_Scc": [0.2513, 0.2172, 0.1939, 0.1771, 0.2364, 0.2032, 0.1974, 0.1746, 0.1109],
        "WP_Nums": [269, 222, 112, 97, 253, 131, 107, 95, 81]
    }
}

import numpy as np
import matplotlib.pyplot as plt

# Style settings
bar_width = 0.4
colors = {"ALL_Scc": "#1f77b4", "WP_Nums": "#ff7f0e"}  # blue & orange
hatch_style = "//"
label_fontsize = 14
tick_fontsize = 12
title_fontsize = 16

# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# Global legend (placed outside the subplots)
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=colors["ALL_Scc"], label="ALL_Scc"),
    plt.Rectangle((0, 0), 1, 1, fc=colors["WP_Nums"], hatch=hatch_style, label="WP_Nums"),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=2,
    fontsize=label_fontsize,
    frameon=False
)

# Iterate over datasets and plot
for idx, (dataset_name, data) in enumerate(datasets.items()):
    ax = axes.flatten()[idx]
    y = np.arange(len(data["Methods"]))

    # Create a second x-axis (top) for WP_Nums
    ax2 = ax.twiny()

    # Horizontal bar charts
    ax.barh(
        y - bar_width / 2,
        data["ALL_Scc"],
        bar_width,
        color=colors["ALL_Scc"],
        alpha=0.9
    )
    ax2.barh(
        y + bar_width / 2,
        data["WP_Nums"],
        bar_width,
        color=colors["WP_Nums"],
        hatch=hatch_style,
        alpha=0.7
    )

    # Y-axis labels
    ax.set_yticks(y)
    ax.set_yticklabels(data["Methods"], fontsize=tick_fontsize)

    # Subplot title: (a), (b), (c), (d)
    ax.set_title(f"({chr(97 + idx)}) {dataset_name}", fontsize=title_fontsize, pad=4, loc="center")

    # Auto axis ranges with margins
    min_scc, max_scc = min(data["ALL_Scc"]), max(data["ALL_Scc"])
    min_wp, max_wp = min(data["WP_Nums"]), max(data["WP_Nums"])

    scc_margin = (max_scc - min_scc) * 0.2
    wp_margin = (max_wp - min_wp) * 0.2

    scc_min = round(min_scc - scc_margin, 2)
    scc_max = round(max_scc + scc_margin, 2)

    # Keep WP axis away from zero for small-count datasets to improve readability
    wp_min = max(10, int(min_wp - wp_margin)) if dataset_name in ["esrd", "crc"] else int(min_wp - wp_margin)
    wp_max = int(max_wp + wp_margin)

    ax.set_xlim([scc_min, scc_max])
    ax2.set_xlim([wp_min, wp_max])

    # Show solid spines for both axes
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle("-")
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linestyle("-")

# Save and show the figure
plt.savefig("../feature_reduction_experiment_final.jpg", format="jpg", dpi=600, bbox_inches="tight")
plt.savefig("../feature_reduction_experiment_final.eps", format="eps", dpi=600, bbox_inches="tight")
plt.show()
