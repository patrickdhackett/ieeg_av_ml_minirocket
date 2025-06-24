import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import f_oneway

# ───────────────────────────────────────────────────────────────────────────────
#  USER-CONFIGURABLE PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
DATA_DIR             = r"E:\data\project_repos\phzhr_turtles_av_ml\data\ppl"
REGION_COUNTS_CSV    = os.path.join(DATA_DIR, "region_subject_counts.csv")
FREQ_CSV             = os.path.join(DATA_DIR, "f1.csv")
FILE_SUFFIX          = "_ppl_iti.mat"
OUTPUT_DIR           = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\17_plots_ppl"
MINIROCKET_CSV_ENC   = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\13_permutationshuffledtrain_iti\Encoding train\13_onevsall_with_permutationontrain_iti.csv"
MINIROCKET_CSV_DP1   = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\13_permutationshuffledtrain_iti\dp1_train\13_onevsall_with_permutationontrain_iti_dptrain.csv"

# Frequency band definitions (Hz)
FREQ_BANDS = {
    "theta":     (4,   8),
    "beta":      (13,  25),
    "low_gamma": (26,  70),
    "high_gamma":(71, 140),
}

# Time windows—must match keys in PPL_data
TIME_WINDOWS = ["Encoding", "DP1", "DP2", "DP3"]

CLASS_LABEL_MAP = {
    1: "Color",
    2: "Orientation",
    3: "Tone",
    4: "Duration"
}

MIN_SUBJ_PER_ROI      = 3   # Only analyze ROI-pairs present in ≥3 subjects
MIN_COMBOS_PER_CLASS = 3   # If a class has < 3 channel-combos, omit its bar

# ───────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def freq_bin_indices(freqs, band):
    """
    Given a sorted 1D array `freqs` and a tuple `(low, high)`,
    return the indices of `freqs` within [low, high].
    """
    low, high = band
    return np.where((freqs >= low) & (freqs <= high))[0]


def load_subject_data(mat_path):
    """
    Load one subject-session .mat that contains:
      - combolabels    (nCombos x 3): [ROI1, ROI2, concatenatedROI]
      - chancombo      (nCombos x 2): bipolar channel indices per combo
      - trialinfo      (nTrials x 8)
      - PPL arrays:    "PPL_fix_all", "PPL_enc_first1k", "PPL_dp_first1k",
                       "PPL_dp_second1k", "PPL_dp_third1k"
      - sigchans1, sigchans2 (each length-4 cell) of bipolar channels
    Returns:
      - combo_labels: numpy array (nCombos, 3)
      - chancombo:    numpy array (nCombos, 2)
      - trialinfo:    numpy array (nTrials, 8)
      - PPL_data:     dict with keys "Fixation","Encoding","DP1","DP2","DP3"
      - sigchans1:    list of length 4, each an array of ints
      - sigchans2:    list of length 4, each an array of ints
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    combo_labels = mat["combolabels"]     # (nCombos, 3) cell
    chancombo    = mat["chancombo"]       # (nCombos, 2) double
    trialinfo    = mat["trialinfo"]       # (nTrials, 8) double

    PPL_data = {
        "Fixation": mat["PPL_fix_all"],       # shape (nTrials, nCombos, nFreqs)
        "Encoding": mat["PPL_enc_first1k"],
        "DP1":      mat["PPL_dp_first1k"],
        "DP2":      mat["PPL_dp_second1k"],
        "DP3":      mat["PPL_dp_third1k"]
    }

    sigchans1 = mat.get("sigchans1", None)  # list of length 4
    sigchans2 = mat.get("sigchans2", None)  # list of length 4

    return combo_labels, chancombo, trialinfo, PPL_data, sigchans1, sigchans2


# ───────────────────────────────────────────────────────────────────────────────
#  PANEL‐PLOTTING FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def plot_panel_all(
    ax, tw_dict_all, subj_all_dict, combo_all_dict, title, r1, r2, fb
):
    """
    Panel #1: 'All Channels' for a given ROI pair & frequency band.

    tw_dict_all[(r1,r2)][fb][tw][cls] -> list of ΔPPL floats
    subj_all_dict[(r1,r2)][fb][tw]       -> set of subjects
    combo_all_dict[(r1,r2)][fb][tw]      -> set of combo indices

    title: string to put above this subplot
    r1, r2, fb: for annotating n_subj / n_combo
    """
    time_order   = ["Encoding", "DP1", "DP2", "DP3"]
    n_windows    = len(time_order)
    class_labels = [1, 2, 3, 4]
    bar_width    = 0.16
    x            = np.arange(n_windows)

    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    # Determine which classes have ≥ MIN_COMBOS_PER_CLASS combos
    invalid_classes = set()
    for cls in class_labels:
        # Union of combos across all valid time windows
        all_combos_cls = set().union(
            *[combo_all_dict[(r1, r2)][fb][tw] for tw in time_order]
        )
        if len(all_combos_cls) < MIN_COMBOS_PER_CLASS:
            invalid_classes.add(cls)

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            arr_list = tw_dict_all[(r1, r2)][fb][tw][cls]
            vals = np.array(arr_list, dtype=float) if (cls not in invalid_classes) else np.array([])
            if vals.size > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means[i, j] = np.nanmean(vals)
                    sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
            pooled_vals[tw].append(vals)

    # Plot bars + faint dots
    for i, cls in enumerate(class_labels):
        if cls in invalid_classes:
            continue
        offset = (i - 1.5) * bar_width
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means[i, :],
            width=bar_width,
            yerr=sems[i, :],
            label=CLASS_LABEL_MAP[cls],
            capsize=3
        )
        for j in range(n_windows):
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_all[(r1, r2)][fb][time_order[j]][cls], dtype=float)
            if vals.size == 0:
                continue
            jitter = np.random.normal(loc=0.0, scale=(bar_width / 6), size=vals.size)
            ax.scatter(
                np.full(vals.shape, x[j] + offset) + jitter,
                vals,
                s=6,
                marker=".",
                color="#555555",
                alpha=0.1
            )

    max_bar = np.nanmax(means + sems) if np.isfinite(means).any() else 0.0

    # Single n_subj, n_combo for ALL classes/time windows
    all_subj = set().union(
        *[subj_all_dict[(r1, r2)][fb].get(tw, set()) for tw in time_order]
    )
    all_combos = set().union(
        *[combo_all_dict[(r1, r2)][fb].get(tw, set()) for tw in time_order]
    )
    n_subj_tot  = len(all_subj)
    n_combo_tot = len(all_combos)
    ax.text(
        0.5, max_bar + 0.08,
        f"n_subj={n_subj_tot}, n_combo={n_combo_tot}",
        ha="center", va="bottom", fontsize=10
    )

    # ANOVA stars per time window
    for j, tw in enumerate(time_order):
        groups = []
        for cls in class_labels:
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_all[(r1, r2)][fb][tw][cls], dtype=float)
            if vals.size > 0:
                groups.append(vals)
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(
                    x[j], max_bar + 0.12,
                    star, ha="center", va="bottom",
                    color="red", fontsize=16
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
    ax.set_xlabel("Time Window")
    ax.set_ylabel("DeltaPPL (Window – Fix)")
    ax.set_title(title)
    ax.legend(title="Class", loc="upper right", fontsize="small")
    ax.axhline(0, color="gray", linewidth=1)


def plot_panel_gamma(
    ax, tw_dict_gamma, subj_gamma_dict, combo_gamma_dict, title, r1, r2
):
    """
    Panel #2: 'Gamma-Only Channels' for a given ROI pair & frequency band.

    tw_dict_gamma[(r1,r2)][fb][tw][cls]  -> list of ΔPPL floats
    subj_gamma_dict[(r1,r2)][cls]        -> set of subjects
    combo_gamma_dict[(r1,r2)][cls]       -> set of combo indices

    We plot a single “n_subj, n_combo” line for each class, 
    (since gamma sets do not vary with time window).
    """
    time_order   = ["Encoding", "DP1", "DP2", "DP3"]
    n_windows    = len(time_order)
    class_labels = [1, 2, 3, 4]
    bar_width    = 0.16
    x            = np.arange(n_windows)

    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    # Determine invalid classes: fewer than MIN_COMBOS_PER_CLASS combos overall
    invalid_classes = set()
    for cls in class_labels:
        if len(combo_gamma_dict[(r1, r2)][cls]) < MIN_COMBOS_PER_CLASS:
            invalid_classes.add(cls)

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            arr_list = tw_dict_gamma[(r1, r2)][fb][tw][cls]
            vals = np.array(arr_list, dtype=float) if (cls not in invalid_classes) else np.array([])
            if vals.size > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means[i, j] = np.nanmean(vals)
                    sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
            pooled_vals[tw].append(vals)

    # Plot bars + very faint dots
    for i, cls in enumerate(class_labels):
        if cls in invalid_classes:
            continue
        offset = (i - 1.5) * bar_width
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means[i, :],
            width=bar_width,
            yerr=sems[i, :],
            label=CLASS_LABEL_MAP[cls],
            capsize=3
        )
        for j in range(n_windows):
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_gamma[(r1, r2)][fb][time_order[j]][cls], dtype=float)
            if vals.size == 0:
                continue
            jitter = np.random.normal(loc=0.0, scale=(bar_width / 6), size=vals.size)
            ax.scatter(
                np.full(vals.shape, x[j] + offset) + jitter,
                vals,
                s=6,
                marker=".",
                color="#444444",
                alpha=0.05
            )

    max_bar = np.nanmax(means + sems) if np.isfinite(means).any() else 0.0

    # For Gamma panel, list "n_subj, n_combo" per class (same across time windows)
    info_lines = []
    for cls in class_labels:
        if cls in invalid_classes:
            continue
        nsubj_cls = len(subj_gamma_dict[(r1, r2)][cls])
        ncombo_cls = len(combo_gamma_dict[(r1, r2)][cls])
        info_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={nsubj_cls}, n_combo={ncombo_cls}")
    textstr = "\n".join(info_lines)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=8, va='top', ha='left'
    )

    # ANOVA stars per time window
    for j, tw in enumerate(time_order):
        groups = []
        for cls in class_labels:
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_gamma[(r1, r2)][fb][tw][cls], dtype=float)
            if vals.size > 0:
                groups.append(vals)
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(
                    x[j], max_bar + 0.12,
                    star, ha="center", va="bottom",
                    color="red", fontsize=16
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
    ax.set_xlabel("Time Window")
    ax.set_title(title)
    ax.legend(title="Class", loc="upper right", fontsize="small")
    ax.axhline(0, color="gray", linewidth=1)


def plot_panel_mr_enc(
    ax, tw_dict_enc, subj_enc_dict, combo_enc_dict, title, r1, r2
):
    """
    Panel #3: 'MiniRocket Encoding-trained Only' channels

    tw_dict_enc[(r1,r2)][fb][tw][cls]  -> list of ΔPPL floats
    subj_enc_dict[(r1,r2)][cls]        -> set of subjects
    combo_enc_dict[(r1,r2)][cls]       -> set of combo indices

    Logic is identical to the gamma panel.
    """
    time_order   = ["Encoding", "DP1", "DP2", "DP3"]
    n_windows    = len(time_order)
    class_labels = [1, 2, 3, 4]
    bar_width    = 0.16
    x            = np.arange(n_windows)

    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    invalid_classes = set()
    for cls in class_labels:
        if len(combo_enc_dict[(r1, r2)][cls]) < MIN_COMBOS_PER_CLASS:
            invalid_classes.add(cls)

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            arr_list = tw_dict_enc[(r1, r2)][fb][tw][cls]
            vals = np.array(arr_list, dtype=float) if (cls not in invalid_classes) else np.array([])
            if vals.size > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means[i, j] = np.nanmean(vals)
                    sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
            pooled_vals[tw].append(vals)

    for i, cls in enumerate(class_labels):
        if cls in invalid_classes:
            continue
        offset = (i - 1.5) * bar_width
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means[i, :],
            width=bar_width,
            yerr=sems[i, :],
            label=CLASS_LABEL_MAP[cls],
            capsize=3
        )
        for j in range(n_windows):
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_enc[(r1, r2)][fb][time_order[j]][cls], dtype=float)
            if vals.size == 0:
                continue
            jitter = np.random.normal(loc=0.0, scale=(bar_width / 6), size=vals.size)
            ax.scatter(
                np.full(vals.shape, x[j] + offset) + jitter,
                vals,
                s=6,
                marker=".",
                color="#444444",
                alpha=0.05
            )

    max_bar = np.nanmax(means + sems) if np.isfinite(means).any() else 0.0

    # One text block: one line per class
    info_lines = []
    for cls in class_labels:
        if cls in invalid_classes:
            continue
        nsubj_cls  = len(subj_enc_dict[(r1, r2)][cls])
        ncombo_cls = len(combo_enc_dict[(r1, r2)][cls])
        info_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={nsubj_cls}, n_combo={ncombo_cls}")
    textstr = "\n".join(info_lines)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=8, va='top', ha='left'
    )

    for j, tw in enumerate(time_order):
        groups = []
        for cls in class_labels:
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_enc[(r1, r2)][fb][tw][cls], dtype=float)
            if vals.size > 0:
                groups.append(vals)
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(
                    x[j], max_bar + 0.12,
                    star, ha="center", va="bottom",
                    color="red", fontsize=16
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
    ax.set_xlabel("Time Window")
    ax.set_title(title)
    ax.legend(title="Class", loc="upper right", fontsize="small")
    ax.axhline(0, color="gray", linewidth=1)


def plot_panel_mr_dp1(
    ax, tw_dict_dp1, subj_dp1_dict, combo_dp1_dict, title, r1, r2
):
    """
    Panel #4: 'MiniRocket DP1-trained Only' channels.

    Same logic as plot_panel_mr_enc, but using the DP1-trained sets.
    """
    time_order   = ["Encoding", "DP1", "DP2", "DP3"]
    n_windows    = len(time_order)
    class_labels = [1, 2, 3, 4]
    bar_width    = 0.16
    x            = np.arange(n_windows)

    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    invalid_classes = set()
    for cls in class_labels:
        if len(combo_dp1_dict[(r1, r2)][cls]) < MIN_COMBOS_PER_CLASS:
            invalid_classes.add(cls)

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            arr_list = tw_dict_dp1[(r1, r2)][fb][tw][cls]
            vals = np.array(arr_list, dtype=float) if (cls not in invalid_classes) else np.array([])
            if vals.size > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means[i, j] = np.nanmean(vals)
                    sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
            pooled_vals[tw].append(vals)

    for i, cls in enumerate(class_labels):
        if cls in invalid_classes:
            continue
        offset = (i - 1.5) * bar_width
        bar_positions = x + offset
        ax.bar(
            bar_positions,
            means[i, :],
            width=bar_width,
            yerr=sems[i, :],
            label=CLASS_LABEL_MAP[cls],
            capsize=3
        )
        for j in range(n_windows):
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_dp1[(r1, r2)][fb][time_order[j]][cls], dtype=float)
            if vals.size == 0:
                continue
            jitter = np.random.normal(loc=0.0, scale=(bar_width / 6), size=vals.size)
            ax.scatter(
                np.full(vals.shape, x[j] + offset) + jitter,
                vals,
                s=6,
                marker=".",
                color="#444444",
                alpha=0.05
            )

    max_bar = np.nanmax(means + sems) if np.isfinite(means).any() else 0.0

    info_lines = []
    for cls in class_labels:
        if cls in invalid_classes:
            continue
        nsubj_cls  = len(subj_dp1_dict[(r1, r2)][cls])
        ncombo_cls = len(combo_dp1_dict[(r1, r2)][cls])
        info_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={nsubj_cls}, n_combo={ncombo_cls}")
    textstr = "\n".join(info_lines)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=8, va='top', ha='left'
    )

    for j, tw in enumerate(time_order):
        groups = []
        for cls in class_labels:
            if cls in invalid_classes:
                continue
            vals = np.array(tw_dict_dp1[(r1, r2)][fb][tw][cls], dtype=float)
            if vals.size > 0:
                groups.append(vals)
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(
                    x[j], max_bar + 0.12,
                    star, ha="center", va="bottom",
                    color="red", fontsize=16
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
    ax.set_xlabel("Time Window")
    ax.set_title(title)
    ax.legend(title="Class", loc="upper right", fontsize="small")
    ax.axhline(0, color="gray", linewidth=1)


# ───────────────────────────────────────────────────────────────────────────────
#  MAIN SCRIPT
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Read region counts (CSV has header: Region, SubjectCount)
    df_regions = pd.read_csv(REGION_COUNTS_CSV)
    df_regions["Region"] = df_regions["Region"].str.strip("[]'")  # "['L ACC']" → "L ACC"
    valid_regions = df_regions.loc[
        df_regions["SubjectCount"] >= MIN_SUBJ_PER_ROI, "Region"
    ].tolist()
    print(f"Valid regions (≥{MIN_SUBJ_PER_ROI} subjects): {len(valid_regions)} found.\n")

    # 2) Generate all unique ROI-pairs
    import itertools
    roi_pairs = list(itertools.combinations(valid_regions, 2))
    print(f"Generated {len(roi_pairs)} ROI-pairs to consider.\n")

    # 3) Read frequency vector from CSV
    freqs = pd.read_csv(FREQ_CSV, header=None).values.ravel()
    print(f"Loaded {len(freqs)} frequency bins.\n")

    # 4) Read MiniRocket CSVs
    df_mr_enc = pd.read_csv(MINIROCKET_CSV_ENC)
    df_mr_enc = df_mr_enc[
        ["Subject", "ChannelIndex", "ClassLabel", "D1_signif_vsFix"]
    ]
    df_mr_dp1 = pd.read_csv(MINIROCKET_CSV_DP1)
    df_mr_dp1 = df_mr_dp1[
        ["Subject", "ChannelIndex", "ClassLabel", "D2_signif_vsFix"]
    ]

    # 5) Initialize nested data structures
    data_all      = {}
    data_gamma    = {}
    data_mr_enc   = {}
    data_mr_dp1   = {}
    subj_all      = {}
    combo_all     = {}
    subj_gamma    = {}
    combo_gamma   = {}
    subj_enc_mr   = {}
    combo_enc_mr  = {}
    subj_dp1_mr   = {}
    combo_dp1_mr  = {}

    for (r1, r2) in roi_pairs:
        # All channels
        data_all[(r1, r2)]   = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_all[(r1, r2)]   = {
            fb: { tw: set() for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        combo_all[(r1, r2)]  = {
            fb: { tw: set() for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }

        # Gamma-only
        data_gamma[(r1, r2)] = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_gamma[(r1, r2)] = { cls: set() for cls in [1, 2, 3, 4] }
        combo_gamma[(r1, r2)] = { cls: set() for cls in [1, 2, 3, 4] }

        # MiniRocket (Encoding-trained)
        data_mr_enc[(r1, r2)]  = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_enc_mr[(r1, r2)]   = { cls: set() for cls in [1, 2, 3, 4] }
        combo_enc_mr[(r1, r2)]  = { cls: set() for cls in [1, 2, 3, 4] }

        # MiniRocket (DP1-trained)
        data_mr_dp1[(r1, r2)]  = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_dp1_mr[(r1, r2)]  = { cls: set() for cls in [1, 2, 3, 4] }
        combo_dp1_mr[(r1, r2)] = { cls: set() for cls in [1, 2, 3, 4] }

    # 6) Loop over subject-session PPL files
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]
    print(f"Found {len(mat_files)} subject-session files.\n")

    for fname in mat_files:
        subj = fname.split("_")[0]
        mat_path = os.path.join(DATA_DIR, fname)
        print(f"Loading file: {fname}")

        combo_labels, chancombo, trialinfo, PPL_data, sigchans1, sigchans2 = \
            load_subject_data(mat_path)

        # 6a) Drop any trial with NaN in trialinfo
        valid_trials_mask = ~np.isnan(trialinfo).any(axis=1)
        nTrials = trialinfo.shape[0]
        nValid  = valid_trials_mask.sum()
        print(f"  Total trials: {nTrials}, Valid trials: {nValid}")
        if nValid == 0:
            print("  → No valid trials; skipping.\n")
            continue

        trialinfo_valid = trialinfo[valid_trials_mask, :]
        trial_classes   = trialinfo_valid[:, 6].astype(int)

        # Filter PPL arrays
        PPL_fix_valid = PPL_data["Fixation"][valid_trials_mask, :, :]
        PPL_enc_valid = PPL_data["Encoding"][valid_trials_mask, :, :]
        PPL_dp1_valid = PPL_data["DP1"][valid_trials_mask, :, :]
        PPL_dp2_valid = PPL_data["DP2"][valid_trials_mask, :, :]
        PPL_dp3_valid = PPL_data["DP3"][valid_trials_mask, :, :]

        nValid, nCombos, nFreqs = PPL_fix_valid.shape

        # Compute ΔPPL
        dPPL = {
            "Encoding": PPL_enc_valid - PPL_fix_valid,
            "DP1":      PPL_dp1_valid - PPL_fix_valid,
            "DP2":      PPL_dp2_valid - PPL_fix_valid,
            "DP3":      PPL_dp3_valid - PPL_fix_valid
        }

        # Build gamma-modulated sets per class (bipolar channel indices)
        gamma_sets = {}
        if (sigchans1 is not None) and (sigchans2 is not None):
            for cls in [1, 2, 3, 4]:
                arr1 = np.atleast_1d(sigchans1[cls-1]).astype(int).ravel()
                arr2 = np.atleast_1d(sigchans2[cls-1]).astype(int).ravel()
                gamma_sets[cls] = set(arr1.tolist() + arr2.tolist())
        else:
            for cls in [1, 2, 3, 4]:
                gamma_sets[cls] = set()

        # Build MiniRocket-encoding‐trained sets per class
        sub_enc = df_mr_enc[df_mr_enc["Subject"] == subj]
        mr_enc_sets = {}
        for cls in [1, 2, 3, 4]:
            cls_df = sub_enc[
                (sub_enc["ClassLabel"] == cls) &
                (sub_enc["D1_signif_vsFix"] == True)
            ]
            mr_enc_sets[cls] = set(cls_df["ChannelIndex"].astype(int).tolist())

        # Build MiniRocket-DP1‐trained sets per class
        sub_dp1 = df_mr_dp1[df_mr_dp1["Subject"] == subj]
        mr_dp1_sets = {}
        for cls in [1, 2, 3, 4]:
            cls_df = sub_dp1[
                (sub_dp1["ClassLabel"] == cls) &
                (sub_dp1["D2_signif_vsFix"] == True)
            ]
            mr_dp1_sets[cls] = set(cls_df["ChannelIndex"].astype(int).tolist())

        # 6b) For each ROI-pair, find matching combos
        for (r1, r2) in roi_pairs:
            concat1 = r1 + r2
            concat2 = r2 + r1
            combo_idxs = [
                i for i in range(nCombos)
                if (str(combo_labels[i][2]) == concat1) or
                   (str(combo_labels[i][2]) == concat2)
            ]
            if not combo_idxs:
                print(f"    ROI-pair ({r1} vs {r2}): 0 combos → skipping.")
                continue

            combo_idxs = np.array(combo_idxs, dtype=int)
            print(f"    ROI-pair ({r1} vs {r2}): {len(combo_idxs)} combos.")

            # Precompute freq-bin indices
            band_indices = {
                fb: freq_bin_indices(freqs, FREQ_BANDS[fb])
                for fb in FREQ_BANDS
            }

            # 6c) For each time window & frequency band, pool ΔPPL per class
            for fb, fidx in band_indices.items():
                if fidx.size == 0:
                    continue
                for tw in TIME_WINDOWS:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        arr_band = np.nanmean(dPPL[tw][:, :, fidx], axis=2)

                    for cls in [1, 2, 3, 4]:
                        trial_idxs = np.where(trial_classes == cls)[0]
                        if trial_idxs.size == 0:
                            continue

                        # --- All Channels ---
                        submat_all = arr_band[np.ix_(trial_idxs, combo_idxs)]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_per_combo = np.nanmean(submat_all, axis=0)
                        vals_all = mean_per_combo[~np.isnan(mean_per_combo)]
                        if vals_all.size > 0:
                            data_all[(r1, r2)][fb][tw][cls].extend(vals_all.tolist())
                            subj_all[(r1, r2)][fb][tw].add(subj)
                            combo_all[(r1, r2)][fb][tw].update(combo_idxs.tolist())

                        # --- Gamma-only Channels ---
                        gamma_set = gamma_sets[cls]
                        valid_gamma = [
                            idx for idx in combo_idxs
                            if (int(chancombo[idx, 0]) in gamma_set) and
                               (int(chancombo[idx, 1]) in gamma_set)
                        ]
                        if valid_gamma:
                            valid_gamma = np.array(valid_gamma, dtype=int)
                            submat_g = arr_band[np.ix_(trial_idxs, valid_gamma)]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                mean_g = np.nanmean(submat_g, axis=0)
                            vals_g = mean_g[~np.isnan(mean_g)]
                            if vals_g.size > 0:
                                data_gamma[(r1, r2)][fb][tw][cls].extend(vals_g.tolist())
                                subj_gamma[(r1, r2)][cls].add(subj)
                                combo_gamma[(r1, r2)][cls].update(valid_gamma.tolist())

                        # --- MiniRocket Encoding-trained Only ---
                        enc_set = mr_enc_sets[cls]
                        valid_enc = [
                            idx for idx in combo_idxs
                            if (int(chancombo[idx, 0]) in enc_set) and
                               (int(chancombo[idx, 1]) in enc_set)
                        ]
                        if valid_enc:
                            valid_enc = np.array(valid_enc, dtype=int)
                            submat_e = arr_band[np.ix_(trial_idxs, valid_enc)]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                mean_e = np.nanmean(submat_e, axis=0)
                            vals_e = mean_e[~np.isnan(mean_e)]
                            if vals_e.size > 0:
                                data_mr_enc[(r1, r2)][fb][tw][cls].extend(vals_e.tolist())
                                subj_enc_mr[(r1, r2)][cls].add(subj)
                                combo_enc_mr[(r1, r2)][cls].update(valid_enc.tolist())

                        # --- MiniRocket DP1-trained Only ---
                        dp1_set = mr_dp1_sets[cls]
                        valid_dp1 = [
                            idx for idx in combo_idxs
                            if (int(chancombo[idx, 0]) in dp1_set) and
                               (int(chancombo[idx, 1]) in dp1_set)
                        ]
                        if valid_dp1:
                            valid_dp1 = np.array(valid_dp1, dtype=int)
                            submat_d = arr_band[np.ix_(trial_idxs, valid_dp1)]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                mean_d = np.nanmean(submat_d, axis=0)
                            vals_d = mean_d[~np.isnan(mean_d)]
                            if vals_d.size > 0:
                                data_mr_dp1[(r1, r2)][fb][tw][cls].extend(vals_d.tolist())
                                subj_dp1_mr[(r1, r2)][cls].add(subj)
                                combo_dp1_mr[(r1, r2)][cls].update(valid_dp1.tolist())

        print("")  # blank line between files

    # ───────────────────────────────────────────────────────────────────────────────
    #  7) Plotting & ANOVA (4 panels: All / Gamma / MR-Enc / MR-DP1)
    # ───────────────────────────────────────────────────────────────────────────────
    total_plots = 0
    for (r1, r2) in roi_pairs:
        for fb in FREQ_BANDS.keys():
            tw_all   = data_all
            tw_gamma = data_gamma
            tw_enc   = data_mr_enc
            tw_dp1   = data_mr_dp1

            # Check if any data at all for this ROI-pair & band
            has_all   = any(
                len(tw_all[(r1, r2)][fb][tw][cls]) > 0
                for tw in TIME_WINDOWS for cls in [1,2,3,4]
            )
            has_gamma = any(
                len(tw_gamma[(r1, r2)][fb][tw][cls]) > 0
                for tw in TIME_WINDOWS for cls in [1,2,3,4]
            )
            has_enc   = any(
                len(tw_enc[(r1, r2)][fb][tw][cls]) > 0
                for tw in TIME_WINDOWS for cls in [1,2,3,4]
            )
            has_dp1   = any(
                len(tw_dp1[(r1, r2)][fb][tw][cls]) > 0
                for tw in TIME_WINDOWS for cls in [1,2,3,4]
            )

            if not (has_all or has_gamma or has_enc or has_dp1):
                print(f"Skipping {r1} vs {r2} | {fb}: no data in any panel.")
                continue

            total_plots += 1
            print(f"Plotting {total_plots}: ROI pair {r1} vs {r2}, Frequency band: {fb}")

            fig, axes = plt.subplots(1, 4, figsize=(28, 6), sharey=True)
            plt.suptitle(f"{r1} vs {r2}  |  {fb} band", fontsize=18)

            # Panel 1: All Channels
            ax0 = axes[0]
            plot_panel_all(
                ax0,
                tw_all,
                subj_all,
                combo_all,
                title="All Channels",
                r1=r1, r2=r2, fb=fb
            )

            # Panel 2: Gamma-Only Channels
            ax1 = axes[1]
            plot_panel_gamma(
                ax1,
                tw_gamma,
                subj_gamma,
                combo_gamma,
                title="Gamma-Only Channels",
                r1=r1, r2=r2
            )

            # Panel 3: MiniRocket (Encoding-trained)
            ax2 = axes[2]
            plot_panel_mr_enc(
                ax2,
                tw_enc,
                subj_enc_mr,
                combo_enc_mr,
                title="MiniRocket Encoding-Trained",
                r1=r1, r2=r2
            )

            # Panel 4: MiniRocket (DP1-trained)
            ax3 = axes[3]
            plot_panel_mr_dp1(
                ax3,
                tw_dp1,
                subj_dp1_mr,
                combo_dp1_mr,
                title="MiniRocket DP1-Trained",
                r1=r1, r2=r2
            )

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            fname_combined = f"{fb}_{r1.replace('/', '_')}_{r2.replace('/', '_')}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname_combined), dpi=150)
            plt.close()

    print(f"\nFinished plotting {total_plots} figures. Check '{OUTPUT_DIR}' for output.")
