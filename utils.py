import matplotlib.pyplot as plt


def perf_plot(labels, values):
    _, ax = plt.subplots(figsize=(5, 3), dpi=150)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    
    ax.set_yscale("log")
    ax.set_title("Matmul Runtime Comparison", fontsize=10, pad=10)
    ax.set_ylabel("Runtime mean (ms, log scale)", fontsize=8)
    ax.set_xlabel("Implementation", fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2, value * 1.1, f"{value:.1f}", 
            ha="center", va="bottom", fontsize=9
        )

    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim([min(values)*0.8, max(values)*5])

    plt.tight_layout()
    plt.show()
