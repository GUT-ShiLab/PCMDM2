import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["MelonnPan", "ENVIM", "MMINP", "MiMeNet", "BiomeNED", "mNODE", "LOCATE", "PCMDM2"]
all_scc = [0.3501, 0.2731, 0.384, 0.234, 0.1497, 0.2487, 0.2213, 0.2822]
wp_nums = [592, 412, 836, 3218, 1636, 3669, 3110, 4176]

# Color settings
bar_color = "#759DCA"   # Main bar color (blue)
line_color = "#519D78"  # Line color (green)
text_color = "#000000"

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart
bars = ax1.bar(methods, all_scc, color=bar_color, alpha=0.8, label="ALL_Scc", width=0.6)

ax1.set_ylabel("ALL_Scc", fontsize=14, color=text_color)
ax1.set_ylim(0, max(all_scc) * 1.3)
ax1.tick_params(axis="y", labelcolor=text_color)

# Annotate bar values
for i, (bar, value) in enumerate(zip(bars, all_scc)):
    # Slight adjustment to avoid overlap for specific method label
    offset = -0.02 if methods[i] == "BiomeNED" else 0.01
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color=text_color,
    )

# Secondary axis: line chart
ax2 = ax1.twinx()
ax2.plot(
    methods,
    wp_nums,
    color=line_color,
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    label="WP_Nums",
    zorder=3,
)

ax2.set_ylabel("WP_Nums", fontsize=14, color=text_color)
ax2.set_ylim(0, max(wp_nums) * 1.2)
ax2.tick_params(axis="y", labelcolor=text_color)

# Annotate line point values
for i, value in enumerate(wp_nums):
    # Adjust offsets to avoid label overlap for certain methods
    y_offset = 350 if methods[i] == "BiomeNED" else 150
    x_offset = -0.04 if methods[i] in ["BiomeNED", "MMINP", "LOCATE"] else 0
    ax2.text(
        i + x_offset,
        value + y_offset,
        f"{value}",
        ha="center",
        va="bottom",
        fontsize=12,
        color=text_color,
    )

# Grid lines
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Axis label
ax1.set_xlabel("Methods", fontsize=14)

# Legends
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

# Layout optimization
plt.tight_layout()

# Save the figure

plt.savefig("../method_comparison.svg", format="svg", dpi=600)

# Show the figure
plt.show()
