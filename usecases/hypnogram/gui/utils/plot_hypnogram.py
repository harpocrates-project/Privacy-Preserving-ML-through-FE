from yasa import plot_hypnogram as _plot_hypnogram
from yasa.hypno import Hypnogram, hypno_int_to_str

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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


def load_hypnogram(file_path):
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

def plot_hypnogram(hypnogram):
    hypnogram = Hypnogram(hypno_int_to_str(hypnogram, mapping_dict={0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R", -1: "Art", 9: "Uns"}))
    _plot_hypnogram(hypnogram, highlight="UNS")


def plot_hypnogram_with_stats(hypnogram, sf_hyp=1/30, save_path=None):
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

    # Save to file if a path is supplied
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        # Add an OK button to close the figure
        axes = plt.axes((0.9, 0.01, 0.06, 0.06))
        btn_ok = Button(axes, "OK")
        btn_ok.on_clicked(lambda event: plt.close())

        plt.show()

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
    hypnogram = load_hypnogram(args.hypnogram_file)
    plot_hypnogram_with_stats(hypnogram, save_path=args.plot_file)
