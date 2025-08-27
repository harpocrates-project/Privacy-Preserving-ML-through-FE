import sqlite3
import pandas as pd

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


def print_size_comparison_summary(fe_sizes, plaintext_sizes):
    """
    Print summary statistics comparing the `size_bytes` columns of the two DataFrames.
    Shows basic descriptive stats for each and the difference (FE - plaintext) per spade_id.
    """
    # Descriptive stats for FE sizes
    fe_desc = fe_sizes["size_bytes"].describe()

    # Descriptive stats for plaintext sizes
    pt_desc = plaintext_sizes["size_bytes"].describe()

    # Merge on spade_id to compute per‑record differences
    merged = pd.merge(
        fe_sizes[["spade_id", "size_bytes"]].rename(columns={"size_bytes": "fe_size"}),
        plaintext_sizes[["spade_id", "size_bytes"]].rename(columns={"size_bytes": "pt_size"}),
        on="spade_id",
        how="inner",
    )
    diff = merged["fe_size"] - merged["pt_size"]
    diff_desc = diff.describe()

    # Output
    print("FE sizes (bytes) summary:")
    print(fe_desc)
    print("\nPlaintext sizes (bytes) summary:")
    print(pt_desc)
    print("\nDifference (FE - Plaintext) summary:")
    print(diff_desc)


def plot_average_size_comparison(fe_df, pt_df):
    """
    Plot the average `size_bytes` of two DataFrames with error bars showing min and max.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute statistics for FE sizes
    fe_mean = fe_df["size_bytes"].mean()
    fe_min = fe_df["size_bytes"].min()
    fe_max = fe_df["size_bytes"].max()

    # Compute statistics for plaintext sizes
    pt_mean = pt_df["size_bytes"].mean()
    pt_min = pt_df["size_bytes"].min()
    pt_max = pt_df["size_bytes"].max()

    # Error values: lower = mean - min, upper = max - mean
    errors = np.array([
        [fe_mean - fe_min, pt_mean - pt_min],
        [fe_max - fe_mean, pt_max - pt_mean]
    ])

    labels = ["FE", "Plaintext"]
    means = [fe_mean, pt_mean]

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=errors, capsize=5, tick_label=labels, color=["steelblue", "salmon"])
    plt.ylabel("Average size (bytes)")
    plt.title("Average storage size")
    plt.tight_layout()
    plt.show()

# Plot the comparison
fe_sizes = load_fe_sizes()
plaintext_sizes = load_plaintext_sizes()
print_size_comparison_summary(fe_sizes, plaintext_sizes)
plot_average_size_comparison(fe_sizes, plaintext_sizes)
