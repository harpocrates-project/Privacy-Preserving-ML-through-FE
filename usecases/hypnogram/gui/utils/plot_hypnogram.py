import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def _load_sleep_stages(file_path):
    """
    Load sleep stages from a text file where each line contains a single integer stage.
    """
    stages = []
    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    stages.append(int(stripped))
                except ValueError:
                    # Skip non‑numeric lines (e.g., headers or comments)
                    continue
    return stages


def _plot_hypnogram(stages, epoch_seconds=30, save_path=None):
    """
    Plot a hypnogram given a list of sleep stage integers.

    Parameters
    ----------
    stages : list[int]
        Sleep stage per epoch.
    epoch_seconds : int, optional
        Duration of each epoch in seconds (default 30 s).
    save_path : str or pathlib.Path, optional
        If provided, the plot will be saved to this path before being shown.
    """
    # Convert epoch index to time in hours
    times_hours = [i * epoch_seconds / 3600.0 for i in range(len(stages))]

    plt.figure(figsize=(12, 4))
    plt.step(times_hours, stages, where="post", linewidth=1.5)
    plt.xlabel("Time (hours)")
    plt.ylabel("Sleep Stage")
    plt.title("Hypnogram")
    plt.ylim(bottom=0)  # Ensure the y‑axis starts at 0

    # Force y‑axis ticks to be whole numbers only
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save to file if a path is supplied
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        # Add an OK button to close the figure
        axes = plt.axes((0.9, 0.01, 0.06, 0.06))
        btn_ok = Button(axes, "OK")
        btn_ok.on_clicked(lambda event: plt.close())

        plt.show()


def plot_hypnogram(file_path, save_path=None):
    stages = _load_sleep_stages(file_path)
    _plot_hypnogram(stages, save_path=save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot Hypnogram"
    )
    parser.add_argument(
        "hypnogram_file",
        help="Path to the hypnogram text file",
    )
    parser.add_argument(
        "--plot_file",
        dest="plot_file",
        default=None,
        help="Optional path to save the generated plot (e.g., output.png). "
             "If omitted, the plot is only shown interactively.",
    )
    args = parser.parse_args()

    plot_hypnogram(args.hypnogram_file, save_path=args.plot_file)
