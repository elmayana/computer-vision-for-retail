import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

palettes = {
    "Modern Blue + Purple + Teal": [
        "#2563EB", "#7C3AED", "#14B8A6", "#0EA5E9",
        "#4F46E5", "#10B981", "#F59E0B", "#F43F5E"
    ],

    "Executive Dashboard": [
        "#1E3A5F", "#2563EB", "#06B6D4", "#0F766E",
        "#16A34A", "#D97706", "#EA580C", "#DC2626"
    ],

    "Purple-First Analytics": [
        "#6D28D9", "#8B5CF6", "#4F46E5", "#2563EB",
        "#06B6D4", "#14B8A6", "#F59E0B", "#F43F5E"
    ],

    "Dark Dashboard": [
        "#3B82F6", "#8B5CF6", "#2DD4BF", "#22C55E",
        "#FACC15", "#FB923C", "#EF4444", "#94A3B8"
    ],

    "Tableau-Inspired": [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#9C755F"
    ],

    "Premium SaaS": [
        "#2563EB", "#4F46E5", "#7C3AED", "#14B8A6",
        "#10B981", "#F59E0B", "#FB7185", "#64748B"
    ]
}

fig_height = len(palettes) * 1.5
fig, ax = plt.subplots(figsize=(14, fig_height))

ax.set_xlim(0, 9)
ax.set_ylim(0, len(palettes))
ax.axis("off")

for row, (name, colors) in enumerate(reversed(list(palettes.items()))):
    y = row

    # Palette title
    ax.text(
        -0.1, y + 0.5,
        name,
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

    # Color blocks
    for i, color in enumerate(colors):
        rect = Rectangle((i, y + 0.1), 1, 0.8, color=color)
        ax.add_patch(rect)

        # Determine text color for readability
        text_color = "white"
        if color.upper() in ["#FACC15", "#EDC948"]:
            text_color = "black"

        ax.text(
            i + 0.5,
            y + 0.5,
            color,
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()