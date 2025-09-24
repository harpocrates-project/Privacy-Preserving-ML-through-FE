import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import sqlite3

plt.rcParams.update({'font.size': 18})

def load_time_files(directory: str) -> pd.DataFrame:
    """
    Load all ``.time`` files from ``directory`` into a pandas DataFrame.

    The expected filename pattern is ``{plaintext_id}.time`` (any ``_`` in the
    stem is considered part of the ``plaintext_id``). Each file should contain
    lines of the form ``<label> <seconds>`` (e.g. ``real 0.10``). The function
    returns a DataFrame where each row corresponds to one file and the time
    labels become columns.

    Parameters
    ----------
    directory:
        Path to the folder that contains the ``.time`` files.

    Returns
    -------
    pandas.DataFrame
        Columns: ``plaintext_id`` plus one column for each distinct time label
        found across the files.
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
        record: Dict[str, Any] = {"plaintext_id": plaintext_id}
        record.update(time_data)
        records.append(record)

    # Build the DataFrame; ensure missing columns are filled with NaN
    df = pd.DataFrame(records)

    # Optional: order columns (plaintext_id, then sorted time labels)
    ordered_cols = ["plaintext_id"] + sorted(all_labels)
    df = df.reindex(columns=ordered_cols)

    return df

def load_fe_sizes(db_path="SIESTA/hypnogram_database.sqlite"):
    """
    Calculates the approximate size in bytes of each row in the `users_cipher` table.
    Returns a pandas DataFrame with columns `id` (the row identifier) and `size_bytes`.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Pull all rows including the internal rowid
    cur.execute("SELECT rowid, * FROM users_cipher")
    rows = cur.fetchall()

    results = []
    for row in rows:
        # Use the explicit `id` column if present, otherwise fall back to the internal rowid
        row_id = row["id"]

        # Approximate size: sum of the byte length of each column's string/bytes representation
        size = 0
        for col in row.keys():
            if col in ("rowid", "id"):
                continue
            value = row[col]
            if value is None:
                continue
            if isinstance(value, (int, float)):
                size += len(str(value).encode("utf-8"))
            elif isinstance(value, bytes):
                size += len(value)
            else:
                size += len(str(value).encode("utf-8"))

        results.append({"spade_id": row_id, "size_bytes": size})

    # Clean up
    conn.close()

    # Return as a pandas DataFrame
    return pd.DataFrame(results)


def load_plaintext_sizes(ids_path="SIESTA/ids.txt", sizes_path="SIESTA/plaintext_sizes.txt"):
    """
    Load the ids and plaintext sizes files and combine them into a single DataFrame.
    Returns a DataFrame with columns: spade_id, plaintext_id, size_bytes.
    """
    # Load ids.txt (spade_id, plaintext_id)
    ids_df = pd.read_csv(ids_path, sep=r'\s+', header=None, names=["spade_id", "plaintext_id"])
    # Load plaintext_sizes.txt (size_bytes, plaintext_id)
    sizes_df = pd.read_csv(sizes_path, sep=r'\s+', header=None, names=["size_bytes", "plaintext_id"])
    # Merge on plaintext_id
    combined_df = pd.merge(ids_df, sizes_df, on="plaintext_id", how="inner")
    # Reorder columns
    combined_df = combined_df[["spade_id", "plaintext_id", "size_bytes"]]
    return combined_df


def compute_speed_df(sizes_df: pd.DataFrame, times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transfer speed (kilobytes per second) for each ``plaintext_id``.
    This function now supports both ``size_kb`` **and** ``size_bytes`` columns
    in ``sizes_df``. If ``size_kb`` is present it is used directly; otherwise the
    ``size_bytes`` column is converted to kilobytes (1 KB = 1024 bytes).

    Parameters
    ----------
    sizes_df : pd.DataFrame
        DataFrame returned by ``load_size_file``. Must contain the column
        ``plaintext_id`` and either ``size_kb`` **or** ``size_bytes``.
    times_df : pd.DataFrame
        DataFrame returned by ``load_time_files``. Must contain the column
        ``plaintext_id`` and at least one numeric time column. If a ``real``
        column is present it is used; otherwise the first numeric column
        (excluding ``plaintext_id``) is used.

    Returns
    -------
    pd.DataFrame
        Columns ``plaintext_id`` and ``speed_kb_per_s`` (kilobytes per second).
    """
    import pandas as pd

    # ---------- Choose time column ----------
    time_col = "real"

    # ---------- Compute mean time per plaintext_id ----------
    avg_times = (
        times_df.groupby("plaintext_id")[time_col]
        .mean()
        .reset_index()
        .rename(columns={time_col: "mean_time"})
    )

    # ---------- Prepare size information ----------
    # Work on a copy to avoid mutating the caller's DataFrame
    sizes = sizes_df.copy()

    if "size_kb" not in sizes.columns:
        if "size_bytes" in sizes.columns:
            # Convert bytes to kilobytes (float division)
            sizes["size_kb"] = sizes["size_bytes"] / 1024.0
        else:
            raise ValueError(
                "sizes_df must contain either a 'size_kb' or 'size_bytes' column."
            )
    # Now we are guaranteed to have a ``size_kb`` column
    size_subset = sizes[["plaintext_id", "size_kb"]]

    # ---------- Merge sizes with average times ----------
    merged = pd.merge(size_subset, avg_times, on="plaintext_id", how="inner")

    # ---------- Compute speed (KB per second) ----------
    merged["speed_kb_per_s"] = merged["size_kb"] / merged["mean_time"]

    # Return only the requested columns as a DataFrame
    return merged[["plaintext_id", "speed_kb_per_s"]]


def plot_time_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Time Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``real`` time of two
    upload‑time DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames produced by ``load_time_files``.  They must contain a column
        named ``real`` or, if that column is absent, any numeric column (excluding
        ``plaintext_id``) will be used as a fallback.
    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import pandas as pd

    def _choose_time_column(df: pd.DataFrame) -> str:
        """Return the column name to use for averaging."""
        if "real" in df.columns:
            return "real"
        # Fallback: first numeric column that is not the identifier
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "plaintext_id"]
        if not numeric_cols:
            raise ValueError("No suitable numeric time column found in the DataFrame.")
        return numeric_cols[0]

    time_col1 = _choose_time_column(df1)
    time_col2 = _choose_time_column(df2)

    avg1 = df1[time_col1].mean()
    avg2 = df2[time_col2].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([0, 1], [avg1, avg2], color=["tab:blue", "tab:orange"], tick_label=[label1, label2])

    # Annotate bars with the average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Time (seconds)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # give some headroom for the annotations
    plt.tight_layout()
    plt.show()


def plot_speed_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Speed Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average ``speed_kb_per_s`` of two
    speed DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames that contain a ``speed_kb_per_s`` column.  The function
        computes the mean of this column for each DataFrame and displays the
        results as two bars.

    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    # Ensure the required column exists
    if "speed_kb_per_s" not in df1.columns or "speed_kb_per_s" not in df2.columns:
        raise ValueError("Both DataFrames must contain a 'speed_kb_per_s' column.")

    # Compute the average speed for each DataFrame
    avg1 = df1["speed_kb_per_s"].mean()
    avg2 = df2["speed_kb_per_s"].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([0, 1], [avg1, avg2], color=["tab:green", "tab:red"], tick_label=[label1, label2])

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Speed (KB/s)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # add headroom for annotations
    plt.tight_layout()
    plt.show()


def plot_storage_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    title: str = "Storage Comparison",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    """
    Plot a side‑by‑side bar chart comparing the average storage size of two
    DataFrames.  The DataFrames may contain either a ``size_kb`` column **or**
    a ``size_bytes`` column.  If only ``size_bytes`` is present it will be
    converted to kilobytes (1 KB = 1024 bytes) for the comparison.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames produced by ``load_size_file`` (or compatible helpers).  They
        must contain either a ``size_kb`` column or a ``size_bytes`` column.
    title : str, optional
        Title of the plot.
    label1, label2 : str, optional
        Labels for the two bars (default ``"Dataset 1"`` and ``"Dataset 2"``).

    Returns
    -------
    None
        The function displays the plot using ``matplotlib.pyplot.show``.
    """
    import matplotlib.pyplot as plt

    # Helper to obtain the average size in kilobytes from a DataFrame
    def _average_size_kb(df: pd.DataFrame) -> float:
        if "size_kb" in df.columns:
            return df["size_kb"].mean()
        elif "size_bytes" in df.columns:
            # Convert bytes to kilobytes on‑the‑fly, without mutating the original DataFrame
            return (df["size_bytes"] / 1024.0).mean()
        else:
            raise ValueError(
                "DataFrames must contain either a 'size_kb' or a 'size_bytes' column."
            )

    # Compute average sizes for each dataset
    avg1 = _average_size_kb(df1)
    avg2 = _average_size_kb(df2)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [0, 1],
        [avg1, avg2],
        color=["tab:purple", "tab:brown"],
        tick_label=[label1, label2],
    )

    # Annotate each bar with its average value (rounded to 2 decimal places)
    for bar, avg in zip(bars, (avg1, avg2)):
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_ylabel("Average Size (KB)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg1, avg2) * 1.2)  # add headroom for annotations
    plt.tight_layout()
    plt.show()


fe_sizes = load_fe_sizes()
plaintext_sizes = load_plaintext_sizes()

fe_upload_times = load_time_files('SIESTA/fe_upload_time')
rest_upload_times = load_time_files('SIESTA/rest_upload_time')
fe_analyst_times = load_time_files('SIESTA/fe_analyst_time')

plaintext_upload_speed = compute_speed_df(plaintext_sizes, rest_upload_times)
fe_upload_speed = compute_speed_df(plaintext_sizes, fe_upload_times)
plaintext_analyst_times = pd.DataFrame(columns=["real"])

plot_storage_comparison(fe_sizes, plaintext_sizes, title="Average Storage Cost per File", label1="FE", label2="Plaintext")
plot_speed_comparison(fe_upload_speed, plaintext_upload_speed, title="Average Upload Speed per File", label1="FE", label2="REST")
plot_time_comparison(fe_analyst_times, plaintext_analyst_times, title="Average Query Response time", label1="FE", label2="")
plot_time_comparison(fe_upload_times, rest_upload_times, title="Average Upload Time per File", label1="FE", label2="")
