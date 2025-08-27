import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

def load_time_files(directory: str) -> pd.DataFrame:
    """
    Load all ``.time`` files from ``directory`` into a pandas DataFrame.

    The expected filename pattern is ``{plaintext_id}_{query_value}.time``.
    If the pattern does not contain an underscore, the whole stem is treated
    as ``plaintext_id`` and ``query_value`` is set to ``None``.

    Each file should contain lines of the form ``<label> <seconds>`` (e.g.
    ``real 0.10``).  The function returns a DataFrame where each row
    corresponds to one file and the time labels become columns.

    Parameters
    ----------
    directory:
        Path to the folder that contains the ``.time`` files.

    Returns
    -------
    pandas.DataFrame
        Columns: ``plaintext_id``, ``query_value`` plus one column for each
        distinct time label found across the files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"'{directory}' is not a valid directory.")

    records: List[Dict[str, Any]] = []
    # Keep track of all encountered time labels so we can ensure a consistent column order
    all_labels: set = set()

    for file_path in dir_path.glob("*.time"):
        # Parse filename
        stem = file_path.stem  # removes the final ".time"
        if "_" in stem:
            plaintext_id, query_value = stem.rsplit("_", 1)
        else:
            plaintext_id, query_value = stem, None

        # Read the file content
        time_data: Dict[str, float] = {}
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    # Skip malformed lines but keep processing the file
                    continue
                label, value_str = parts
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                time_data[label] = value
                all_labels.add(label)

        # Combine parsed info into a single record
        record: Dict[str, Any] = {
            "plaintext_id": plaintext_id,
            "query_value": query_value,
        }
        record.update(time_data)
        records.append(record)

    # Build the DataFrame; ensure missing columns are filled with NaN
    df = pd.DataFrame(records)

    # Optional: order columns (plaintext_id, query_value, then sorted time labels)
    ordered_cols = ["plaintext_id", "query_value"] + sorted(all_labels)
    df = df.reindex(columns=ordered_cols)

    return df

def plot_time_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str = "Group 1",
    label2: str = "Group 2",
    title: str = "Average Times",
    xlabel: str = "Metric",
    ylabel: str = "Time (seconds)",
) -> None:
    """Plot a grouped bar chart of average 'real' and combined 'user+sys' times for two DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames containing columns ``real``, ``sys`` and ``user`` with time measurements.
    label1, label2 : str
        Labels for the two groups (used in the legend).
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    """
    import numpy as np

    # Metrics to plot: real time and the sum of user and sys times
    metrics = ["real", "user+sys"]
    colors = {"df1": "steelblue", "df2": "orange"}

    # Helper to compute mean and std for each metric
    def compute_stats(df: pd.DataFrame):
        stats: Dict[str, Dict[str, float]] = {}
        # Real time
        if "real" in df.columns:
            series = df["real"].dropna()
            stats["real"] = {"mean": series.mean(), "std": series.std()}
        # Combined user+sys time
        if "sys" in df.columns and "user" in df.columns:
            combined = df["sys"] + df["user"]
            combined = combined.dropna()
            stats["user+sys"] = {"mean": combined.mean(), "std": combined.std()}
        return stats

    stats1 = compute_stats(df1)
    stats2 = compute_stats(df2)

    # Prepare data for plotting
    n_metrics = len(metrics)
    index = np.arange(n_metrics)  # the label locations
    bar_width = 0.35

    means1 = [stats1[m]["mean"] if m in stats1 else np.nan for m in metrics]
    stds1 = [stats1[m]["std"] if m in stats1 else np.nan for m in metrics]

    means2 = [stats2[m]["mean"] if m in stats2 else np.nan for m in metrics]
    stds2 = [stats2[m]["std"] if m in stats2 else np.nan for m in metrics]

    _, ax = plt.subplots(figsize=(8, 6))

    # Group 1 bars
    ax.bar(
        index - bar_width / 2,
        means1,
        bar_width,
        yerr=stds1,
        capsize=5,
        label=label1,
        color=colors["df1"],
        edgecolor="black",
    )

    # Group 2 bars
    ax.bar(
        index + bar_width / 2,
        means2,
        bar_width,
        yerr=stds2,
        capsize=5,
        label=label2,
        color=colors["df2"],
        edgecolor="black",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.show()

fe_upload_times = load_time_files('SIESTA/fe_upload_time')
rest_upload_times = load_time_files('SIESTA/rest_upload_time')
fe_analyst_times = load_time_files('SIESTA/rest_upload_time')
plaintext_analyst_times = pd.DataFrame(columns=["real", "user+sys"])

plot_time_comparison(fe_upload_times, rest_upload_times, title="Average Upload Time", label1="SPADE", label2="REST")
plot_time_comparison(fe_analyst_times, plaintext_analyst_times, title="Average Analyse Time", label1="SPADE", label2="Plaintext")
