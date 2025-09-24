import ipdb

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sleep_statistics(hypno, sf_hyp):
    """Compute standard sleep statistics from an hypnogram.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    hypno : array_like
        Hypnogram, assumed to be already cropped to time in bed (TIB,
        also referred to as Total Recording Time,
        i.e. "lights out" to "lights on").

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
            - 9 = Unscored
    sf_hyp : float
        The sampling frequency of the hypnogram. Should be 1/30 if there is one
        value per 30-seconds, 1/20 if there is one value per 20-seconds,
        1 if there is one value per second, and so on.

    Returns
    -------
    stats : dict
        Sleep statistics (expressed in minutes)

    Notes
    -----
    All values except SE, SME and percentages of each stage are expressed in
    minutes. YASA follows the AASM guidelines to calculate these parameters:

    * Time in Bed (TIB): total duration of the hypnogram.
    * Sleep Period Time (SPT): duration from first to last period of sleep.
    * Wake After Sleep Onset (WASO): duration of wake periods within SPT.
    * Total Sleep Time (TST): total duration of N1 + N2 + N3 + REM sleep in SPT.
    * Sleep Efficiency (SE): TST / TIB * 100 (%).
    * Sleep Maintenance Efficiency (SME): TST / SPT * 100 (%).
    * W, N1, N2, N3 and REM: sleep stages duration. NREM = N1 + N2 + N3.
    * % (W, ... REM): sleep stages duration expressed in percentages of TST.
    * Latencies: latencies of sleep stages from the beginning of the record.
    * Sleep Onset Latency (SOL): Latency to first epoch of any sleep.

    .. warning::
        The definition of REM latency in the AASM scoring manual differs from the REM latency
        reported here. The former uses the time from first epoch of sleep, while YASA uses the
        time from the beginning of the recording. The AASM definition of the REM latency can be
        found with `Lat_REM - SOL`.

    References
    ----------
    * Iber, C. (2007). The AASM manual for the scoring of sleep and
      associated events: rules, terminology and technical specifications.
      American Academy of Sleep Medicine.

    * Silber, M. H., Ancoli-Israel, S., Bonnet, M. H., Chokroverty, S.,
      Grigg-Damberger, M. M., Hirshkowitz, M., Kapen, S., Keenan, S. A.,
      Kryger, M. H., Penzel, T., Pressman, M. R., & Iber, C. (2007).
      `The visual scoring of sleep in adults
      <https://www.ncbi.nlm.nih.gov/pubmed/17557422>`_. Journal of Clinical
      Sleep Medicine: JCSM: Official Publication of the American Academy of
      Sleep Medicine, 3(2), 121–131.
    """
    # warnings.warn(
    #     "The `yasa.sleep_statistics` function is deprecated and will be removed in v0.8. "
    #     "Please use the `yasa.Hypnogram.sleep_statistics` method instead.",
    #     FutureWarning,
    # )
    stats = {}
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, "hypno must have only one dimension."
    assert hypno.size > 1, "hypno must have at least two elements."

    # TIB, first and last sleep
    stats["TIB"] = len(hypno)
    first_sleep = np.where(hypno > 0)[0][0]
    last_sleep = np.where(hypno > 0)[0][-1]

    # Crop to SPT
    hypno_s = hypno[first_sleep : (last_sleep + 1)]
    stats["SPT"] = hypno_s.size
    stats["WASO"] = hypno_s[hypno_s == 0].size
    stats["TST"] = hypno_s[hypno_s > 0].size

    # Duration of each sleep stages
    stats["N1"] = hypno[hypno == 1].size
    stats["N2"] = hypno[hypno == 2].size
    stats["N3"] = hypno[hypno == 3].size
    stats["REM"] = hypno[hypno == 4].size
    stats["NREM"] = stats["N1"] + stats["N2"] + stats["N3"]

    # Sleep stage latencies -- only relevant if hypno is cropped to TIB
    stats["SOL"] = first_sleep
    stats["Lat_N1"] = np.where(hypno == 1)[0].min() if 1 in hypno else np.nan
    stats["Lat_N2"] = np.where(hypno == 2)[0].min() if 2 in hypno else np.nan
    stats["Lat_N3"] = np.where(hypno == 3)[0].min() if 3 in hypno else np.nan
    stats["Lat_REM"] = np.where(hypno == 4)[0].min() if 4 in hypno else np.nan

    # Convert to minutes
    for key, value in stats.items():
        stats[key] = value / (60 * sf_hyp)

    # Percentage
    stats["%N1"] = 100 * stats["N1"] / stats["TST"]
    stats["%N2"] = 100 * stats["N2"] / stats["TST"]
    stats["%N3"] = 100 * stats["N3"] / stats["TST"]
    stats["%REM"] = 100 * stats["REM"] / stats["TST"]
    stats["%NREM"] = 100 * stats["NREM"] / stats["TST"]
    stats["SE"] = 100 * stats["TST"] / stats["TIB"]
    stats["SME"] = 100 * stats["TST"] / stats["SPT"]
    return stats


vifasom_event_to_int_map = {
    'Éveil': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'REM': 4,

    'A. Centrale': -1,
    'A. Mixte': -1,
    'A. Obstructive': -1,
    'Activity-BNend': -1,
    'Activity-BNstart': -1,
    'Activity-CLASSICend': -1,
    'Activity-CLASSICstart': -1,
    'Activity-ENend': -1,
    'Activity-ENstart': -1,
    'Activity-REMend': -1,
    'Activity-REMstart': -1,
    'Avertissement': -1,
    "Début de l'analyse": -1,
    'Désat': -1,
    'Hypopnée': -1,
    'MJ': -1,
    'Micro-éveil': -1,
    'Notification': -1,
    'PLM': -1,
    'RERA': -1,
    'Rapide': -1,
    '[]': -1,
}

zero_query = {
    'Éveil': 0,
    'N1': 9,
    'N2': 9,
    'N3': 9,
    'REM': 9,
}


def plot_hypnogram(hypnogram):
    from yasa import plot_hypnogram
    from yasa.hypno import Hypnogram, hypno_int_to_str
    hypnogram = Hypnogram(hypno_int_to_str(hypnogram, mapping_dict={0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", 9: "Uns"}))
    plot_hypnogram(hypnogram, highlight="UNS")


def load_csv_to_dataframe(file_path, event_to_int_map=vifasom_event_to_int_map):
    # Load only the first four columns
    df = pd.read_csv(file_path, encoding='utf-16', usecols=range(4))
    # Rename columns to English
    df.columns = ['start_time', 'end_time', 'event', 'duration']
    # Remove non‑hypnogram events
    df = df[df['duration'].astype(str) == '30']
    # Add integer mapping for events
    df['event_int'] = df['event'].map(event_to_int_map).fillna(-1).astype(int)
    return df


def load_csvs_to_dataframes(directory, event_to_int_map=vifasom_event_to_int_map):
    dataframe_dict = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # Use the helper to load and process a single CSV file
            df = load_csv_to_dataframe(file_path, event_to_int_map)
            key = os.path.splitext(filename)[0]
            dataframe_dict[key] = df

    return dataframe_dict


def calculate_sleep_statistics(dataframe_dict, sf_hyp=1/30):
    rows = []
    for key, df in dataframe_dict.items():
        # Extract hypnogram as integer stages
        hypno = df['event_int'].to_numpy()
        stats = sleep_statistics(hypno, sf_hyp)
        row = {'id': key}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def combine_sleep_statistics(*dfs, labels=None):
    rows = []
    for i, df in enumerate(dfs):
        # Compute column‑wise mean for numeric columns only
        avg_series = df.mean(numeric_only=True)
        row = avg_series.to_dict()
        # Assign identifier for the source DataFrame
        if labels is not None:
            row["id"] = labels[i]
        else:
            row["id"] = f"df_{i + 1}"
        rows.append(row)
    return pd.DataFrame(rows)

def export_event_int_to_text(dataframe_dict, output_dir):
    import os

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for key, df in dataframe_dict.items():
        # Build the filename using the key
        filename = f"{key}.txt"
        file_path = os.path.join(output_dir, filename)

        # Extract the event_int column as a list of strings
        event_int_series = df['event_int'].astype(str)

        # Write each value on a new line
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(event_int_series.tolist()))


def plot_hypnogram_with_stats(hypnogram, sf_hyp=1/30):
    """
    Plot a hypnogram using YASA's ``plot_hypnogram`` and add a table
    summarising key sleep statistics.

    Parameters
    ----------
    hypnogram : array_like
        Integer hypnogram (e.g., output from ``load_csv_to_dataframe``).
    sf_hyp : float, optional
        Sampling frequency of the hypnogram (default is 1/30 Hz,
        i.e., one epoch per 30 seconds).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the hypnogram plot and the statistics table.
    """
    # Compute sleep statistics
    stats = sleep_statistics(hypnogram, sf_hyp)

    # Prepare a list of rows for the table
    table_rows = [
        ["TIB (min)", f"{stats['TIB']:.2f}"],
        ["SPT (min)", f"{stats['SPT']:.2f}"],
        ["WASO (min)", f"{stats['WASO']:.2f}"],
        ["TST (min)", f"{stats['TST']:.2f}"],
        ["SOL (min)", f"{stats['SOL']:.2f}"],
        ["SE (%)", f"{stats['SE']:.2f}"],
        ["SME (%)", f"{stats['SME']:.2f}"],
    ]

    # Create the figure and axis for the hypnogram
    import matplotlib.pyplot as plt

    # Plot hypnogram
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_hypnogram(hypnogram)

    # Add table underneath the hypnogram
    # Position the table in axis coordinates (0, -0.35) to appear below the plot
    table = ax.table(
        cellText=table_rows,
        colLabels=None,
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.35, 1.0, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Adjust layout so everything fits
    plt.tight_layout()
    return fig


normal_hypnograms = load_csvs_to_dataframes('./VIFASOM/plaintext_data/normal')
insomnia_hypnograms = load_csvs_to_dataframes('./VIFASOM/plaintext_data/insomnia')
sleep_misperception_insomnia_hypnograms = load_csvs_to_dataframes('./VIFASOM/plaintext_data/sleep misperception insomnia')

normal_hypnograms_zero_query = load_csvs_to_dataframes('./VIFASOM/plaintext_data/normal', event_to_int_map=zero_query)
insomnia_hypnograms_zero_query = load_csvs_to_dataframes('./VIFASOM/plaintext_data/insomnia', event_to_int_map=zero_query)
sleep_misperception_insomnia_hypnograms_zero_query = load_csvs_to_dataframes('./VIFASOM/plaintext_data/sleep misperception insomnia', event_to_int_map=zero_query)

# plot_hypnogram_with_stats(insomnia_hypnograms_zero_query['20181031T005107']['event_int'])
normal_sleep_statistics = calculate_sleep_statistics(normal_hypnograms)
insomnia_sleep_stvatistics = calculate_sleep_statistics(insomnia_hypnograms)
sleep_misperception_insomnia_sleep_statistics = calculate_sleep_statistics(sleep_misperception_insomnia_hypnograms)

normal_sleep_statistics_zero_query = calculate_sleep_statistics(normal_hypnograms_zero_query)
insomnia_sleep_statistics_zero_query = calculate_sleep_statistics(insomnia_hypnograms_zero_query)
sleep_misperception_insomnia_sleep_statistics_zero_query = calculate_sleep_statistics(sleep_misperception_insomnia_hypnograms_zero_query)

# ----------------------------------------------------------------------
# Compute and visualise average sleep statistics for each group
# ----------------------------------------------------------------------
# Average statistics for the original hypnograms
avg_stats = combine_sleep_statistics(
    normal_sleep_statistics,
    insomnia_sleep_stvatistics,
    sleep_misperception_insomnia_sleep_statistics,
    labels=[
        "Normal",
        "Insomnia",
        "Sleep Misperception Insomnia",
    ],
)

# Average statistics for the zero‑query hypnograms
avg_stats_zero = combine_sleep_statistics(
    normal_sleep_statistics_zero_query,
    insomnia_sleep_statistics_zero_query,
    sleep_misperception_insomnia_sleep_statistics_zero_query,
    labels=[
        "Normal (Query 0)",
        "Insomnia (Query 0)",
        "Sleep Misperception Insomnia (Query 0)",
    ],
)

# Select the statistics we want to visualise
_stats_to_plot = [
    "TIB",
    "SPT",
    "WASO",
    "TST",
    "SOL",
    "SE",
    "SME",
]

def _plot_average_statistics(df, title):
    """
    Helper to plot a bar chart of the average sleep statistics contained in ``df``.
    """
    # Re‑arrange so that each statistic is a row and each group is a column
    plot_df = df.set_index("id")[_stats_to_plot].transpose()

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        colormap="tab10",
    )
    ax.set_xlabel("Sleep Statistic")
    ax.set_ylabel("Average Value")
    ax.set_title(title)
    ax.legend(title="Group")

    # Add the numeric value above each bar
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            # Position the text slightly above the bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

# Plot the averages for the three original groups
_plot_average_statistics(avg_stats, "Average Sleep Statistics per Group")

# Plot the averages for the three zero‑query groups
_plot_average_statistics(avg_stats_zero, "Average Sleep Statistics per Group (Query 0)")
export_event_int_to_text(normal_hypnograms_zero_query, './VIFASOM/query_results/normal')
export_event_int_to_text(insomnia_hypnograms_zero_query, './VIFASOM/query_results/insomnia')
export_event_int_to_text(sleep_misperception_insomnia_hypnograms_zero_query, './VIFASOM/query_results/sleep misperception insomnia')
