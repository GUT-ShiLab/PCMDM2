import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ------------------ Example data: IBD ------------------

data_ibd = [
    ("[128, 64]", "[32, 64]", 0.32141965188306443),
    ("[256, 128]", "[32, 64]", 0.3500487147347125),
    ("[512, 256]", "[32, 64]", 0.3597960903422278),
    ("[128, 64]", "[64, 128]", 0.3197645785689077),
    ("[256, 128]", "[64, 128]", 0.34656267589754075),
    ("[512, 256]", "[64, 128]", 0.3430201253732272),
    ("[128, 64]", "[128, 256]", 0.3027351748378816),
    ("[256, 128]", "[128, 256]", 0.3197267451796634),
    ("[512, 256]", "[128, 256]", 0.3446154375556828),
    ("[128, 64]", "[16, 32, 64]", 0.3194187036596979),
    ("[256, 128]", "[16, 32, 64]", 0.3490465868153766),
    ("[512, 256]", "[16, 32, 64]", 0.3474888004032911),
    ("[128, 64]", "[32, 64, 128]", 0.3265512008549695),
    ("[256, 128]", "[32, 64, 128]", 0.3501718231067174),
    ("[512, 256]", "[32, 64, 128]", 0.3794),
    ("[128, 64]", "[64, 128, 256]", 0.315630237535856),
    ("[256, 128]", "[64, 128, 256]", 0.33079869176268034),
    ("[512, 256]", "[64, 128, 256]", 0.34889477708018035),
    ("[128, 64]", "[16, 32, 64, 128]", 0.315228625991274),
    ("[256, 128]", "[16, 32, 64, 128]", 0.33921227519423414),
    ("[512, 256]", "[16, 32, 64, 128]", 0.3292718808362162),
    ("[128, 64]", "[32, 64, 128, 256]", 0.3238277835561149),
    ("[256, 128]", "[32, 64, 128, 256]", 0.3396674994619367),
    ("[512, 256]", "[32, 64, 128, 256]", 0.32445476664651424),
]


def parse_conv_dims(conv_str):
    """
    Parse a string like '[16, 32, 64]' into an integer list and return a sorting key:
    key = (number_of_layers, ascending_channel_values)
    """
    nums = re.findall(r"\d+", conv_str)  # extract all numbers
    nums = list(map(int, nums))
    return (len(nums), nums)


def plot_param_heatmap(data, ax, title):
    """
    Convert (mlp_dims, conv_dims, ALL_SCC) tuples into a DataFrame and plot a grouped bar chart.
    """
    df = pd.DataFrame(data, columns=["mlp_dims", "conv_dims", "ALL_SCC"])

    # Use conv_dims as rows, mlp_dims as columns, and ALL_SCC as values
    pivot_df = df.pivot(index="conv_dims", columns="mlp_dims", values="ALL_SCC")

    # Reorder rows (conv_dims) using a custom sorting rule
    sorted_index = sorted(pivot_df.index, key=parse_conv_dims)
    pivot_df = pivot_df.reindex(sorted_index)

    colors = ["#fbb4ae", "#b3cde4", "#cceac4"]
    pivot_df.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("conv_dims", fontsize=13)
    ax.set_ylabel("ALL_SCC", fontsize=13)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    # Draw a dashed horizontal line at the maximum ALL_SCC value
    max_val = df["ALL_SCC"].max()
    ax.axhline(
        y=max_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Max: {max_val:.3f}"
    )

    # Build a clean legend (only once)
    handles, labels = ax.get_legend_handles_labels()

    # Add a prefix to mlp_dims legend entries (keep "Max: xxx" unchanged)
    new_labels = []
    for label in labels:
        if label.startswith("["):  # treat as mlp_dims entry
            new_labels.append(f"mlp_dims: {label}")
        else:
            new_labels.append(label)

    ax.legend(
        handles,
        new_labels,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=12,
        title_fontsize=12
    )

    plt.tight_layout()


# ----------------- Main script -----------------
if __name__ == "__main__":
    # Create a single subplot for the IBD dataset
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # IBD dataset
    plot_param_heatmap(data_ibd, ax, title="IBD Dataset")

    plt.tight_layout()
    plt.savefig("conv_mlp_param_search.svg", dpi=600, format="svg")
    plt.savefig("conv_mlp_param_search.eps", dpi=600, format="eps")
    # plt.savefig("conv_mlp_param_search.emf", dpi=600, format="emf", bbox_inches="tight")
    plt.show()
