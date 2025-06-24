import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# ───────────────────────────────────────────────────────────────────────────────
#  USER-CONFIGURABLE PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
DATA_DIR          = r"E:\data\project_repos\phzhr_turtles_av_ml\data\ppl"
REGION_COUNTS_CSV = os.path.join(DATA_DIR, "region_subject_counts.csv")
FREQ_CSV          = os.path.join(DATA_DIR, "f1.csv")
FILE_SUFFIX       = "_ppl_iti.mat"

MR_ENCODING_CSV   = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\13_permutationshuffledtrain_iti\Encoding train\13_onevsall_with_permutationontrain_iti.csv"
MR_DP1_CSV        = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\13_permutationshuffledtrain_iti\dp1_train\13_onevsall_with_permutationontrain_iti_dptrain.csv"

OUTPUT_DIR        = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\17_plots_ppl"

# Frequency band definitions
FREQ_BANDS = {
    "theta":     (4,   8),
    "beta":      (13,  25),
    "low_gamma": (26,  70),
    "high_gamma":(71, 140),
}

# Time windows—keys must match the keys in the PPL .mat
TIME_WINDOWS = ["Encoding", "DP1", "DP2", "DP3"]

CLASS_LABEL_MAP = {
    1: "Color",
    2: "Orientation",
    3: "Tone",
    4: "Duration"
}

MIN_SUBJ_PER_ROI = 3  # only analyze region pairs present in ≥3 subjects

# ───────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def freq_bin_indices(freqs, band):
    """
    Given a sorted 1D array `freqs` and a tuple (low, high),
    return the indices of freqs within [low, high].
    """
    low, high = band
    return np.where((freqs >= low) & (freqs <= high))[0]

def read_minirocket_csv(csv_path, p_field):
    """
    Read a MiniRocket CSV and return a dict:
      { (subject, class_label) : set(channel_indices) }
    where `p_field` is the column name (e.g. "D1_signif_vsFix" or "D2_signif_vsFix")
    that we filter to True.
    """
    df = pd.read_csv(csv_path)
    df = df[df[p_field] == True]
    out = {}
    for _, row in df.iterrows():
        subj = str(row["Subject"])
        cls  = int(row["ClassLabel"])
        chan = int(row["ChannelIndex"])
        key  = (subj, cls)
        if key not in out:
            out[key] = set()
        out[key].add(chan)
    return out

def load_subject_data(mat_path):
    """
    Load one subject-session .mat. Returns:
      - combo_labels   : numpy array (nCombos, 3) of ROI strings
      - trialinfo      : numpy array (nTrials, 8)
      - PPL_data       : dict { "Fixation","Encoding","DP1","DP2","DP3" }
                          each shape (nTrials, nCombos, nFreqs)
      - sigchans1, sigchans2: length-4 lists, each an ndarray of gamma-modulated channel indices
      - chancombo      : ndarray (nCombos, 2) of bipolarAnat row indices
      - bipolarAnat    : a numpy‐compatible representation such that
                         bipolarAnat[r, 1] is the original channel index (1-based)
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    combo_labels = mat["combolabels"]        # (nCombos, 3) cell of ROI strings
    trialinfo    = mat["trialinfo"]         # (nTrials, 8) double

    # PPL arrays:
    PPL_data = {
        "Fixation": mat["PPL_fix_all"],        # (nTrials, nCombos, nFreqs)
        "Encoding": mat["PPL_enc_first1k"],
        "DP1":      mat["PPL_dp_first1k"],
        "DP2":      mat["PPL_dp_second1k"],
        "DP3":      mat["PPL_dp_third1k"]
    }

    # Gamma-modulated channels per class (1–4), two arrays from MAT:
    sigchans1 = mat.get("sigchans1", [np.array([], dtype=int)] * 4)
    sigchans2 = mat.get("sigchans2", [np.array([], dtype=int)] * 4)

    # Load bipolarAnat and chancombo:
    bipolarAnat = mat["bipolarAnat"]   # MATLAB table-like; column 1=orig idx, column 2=orig channel index (1-based)
    # Convert to a numpy array of original channel indices: e.g. bipolarAnat_orig[r] = int(bipolarAnat[r][1])
    bipolarAnat_orig = np.array([int(row[1].item()) for row in bipolarAnat], dtype=int)

    chancombo = mat["chancombo"]       # shape (nCombos, 2), bipolarAnat row indices (1-based from MATLAB)

    return combo_labels, trialinfo, PPL_data, sigchans1, sigchans2, chancombo, bipolarAnat_orig

def check_any_significant(tw_dict):
    """
    Given tw_dict = { tw -> { cls -> list_of_values } },
    return True if any time window yields p<0.05 by one-way ANOVA.
    """
    for tw in TIME_WINDOWS:
        group_vals = [np.array(tw_dict[tw][cls], dtype=float)
                      for cls in [1,2,3,4] if len(tw_dict[tw][cls]) > 0]
        if len(group_vals) >= 2:
            _, p = f_oneway(*group_vals)
            if p < 0.05:
                return True
    return False

def plot_panel_all(ax, tw_dict, subj_dict, combo_dict):
    """
    Panel 1: All Channels plot (bars + tiny dots).
      - tw_dict:    { tw -> { cls -> [float deltaPPLs] } }
      - subj_dict:  { tw -> set(subjects) }
      - combo_dict: { tw -> set(combo_indices) }
    """
    # Union across all time windows:
    all_subj   = set().union(*(subj_dict[tw] for tw in TIME_WINDOWS))
    all_combos = set().union(*(combo_dict[tw] for tw in TIME_WINDOWS))
    n_subj = len(all_subj)
    n_combo= len(all_combos)

    time_order   = TIME_WINDOWS
    n_windows    = len(time_order)
    class_labels = [1,2,3,4]
    bar_width    = 0.2
    x            = np.arange(n_windows)

    # Compute means and SEMs per class/time window
    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            vals = np.array(tw_dict[tw][cls], dtype=float)
            if vals.size > 0:
                means[i, j] = np.nanmean(vals)
                sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
                pooled_vals[tw].append(vals)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
                pooled_vals[tw].append(np.array([], dtype=float))

    # Plot bars + dots
    for i, cls in enumerate(class_labels):
        offset = (i - 1.5) * bar_width
        for j, tw in enumerate(time_order):
            # Bar
            ax.bar(
                x[j] + offset,
                means[i, j],
                width=bar_width,
                yerr=sems[i, j],
                label=CLASS_LABEL_MAP[cls] if j == 0 else "_nolegend_",
                capsize=4
            )
            # One dot PER channelcombo at the class mean
            n_c = len(combo_dict[tw])
            if n_c > 0:
                dot_positions = np.full(n_c, means[i, j])
                jitter = np.random.normal(loc=0.0, scale=bar_width/10, size=n_c)
                ax.scatter(
                    np.full(dot_positions.shape, x[j] + offset) + jitter,
                    dot_positions,
                    s=4,
                    color="lightgrey",
                    alpha=0.4,
                    marker="."
                )

    # ANOVA stars
    for j, tw in enumerate(time_order):
        group_vals = [pv for pv in pooled_vals[tw] if pv.size > 0]
        if len(group_vals) >= 2:
            _, p = f_oneway(*group_vals)
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            if stars:
                col_heights = means[:, j] + sems[:, j]
                max_bar = np.nanmax(col_heights)
                y_star = max_bar + 0.02
                ax.text(x[j], y_star, stars, ha="center", va="bottom", color="red", fontsize=12)

    # n_subj & n_combo text (top-left)
    ax.text(0.02, 0.95, f"n_subj={n_subj}\nn_combo={n_combo}",
            transform=ax.transAxes, ha="left", va="top", fontsize="8")

    ax.set_xticks(x)
    ax.set_xticklabels(["Enc", "D1", "D2", "D3"], rotation=0)
    ax.set_ylabel("ΔPPL")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(title="Class", loc="upper right", fontsize="6")
    ax.set_title("All Channels", fontsize=10)

def plot_panel_gamma(ax, tw_dict, subj_gamma_dict, combo_gamma_dict, bipolarAnat_orig):
    """
    Panel 2: Gamma-Only Channels.
      - tw_dict:  { tw -> { cls -> [float deltaPPLs] } }
      - subj_gamma_dict:  { (r1,r2) -> { cls -> set(subjects) } }
      - combo_gamma_dict: { (r1,r2) -> { cls -> set(combo_indices) } }
    Note: We’ll have precomputed subj_gamma_dict[(r1,r2)][cls] and combo_gamma_dict[(r1,r2)][cls].
    """
    time_order   = TIME_WINDOWS
    n_windows    = len(time_order)
    class_labels = [1,2,3,4]
    bar_width    = 0.2
    x            = np.arange(n_windows)

    # Compute means and SEMs per class/time window
    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    # We’ll need to know bip_orig only to count combos correctly—but we already have
    # combo counts in combo_gamma_dict[(r1,r2)][cls].
    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            vals = np.array(tw_dict[tw][cls], dtype=float)
            if vals.size > 0:
                means[i, j] = np.nanmean(vals)
                sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
                pooled_vals[tw].append(vals)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
                pooled_vals[tw].append(np.array([], dtype=float))

    # n_subj & n_combo per class (Gamma-only)
    n_subj_cls  = { cls: len(subj_gamma_dict[cls])   for cls in class_labels }
    n_combo_cls = { cls: len(combo_gamma_dict[cls])  for cls in class_labels }

    # Plot bars + points
    for i, cls in enumerate(class_labels):
        offset = (i - 1.5) * bar_width
        for j, tw in enumerate(time_order):
            ax.bar(
                x[j] + offset,
                means[i, j],
                width=bar_width,
                yerr=sems[i, j],
                label=CLASS_LABEL_MAP[cls] if j == 0 else "_nolegend_",
                capsize=4
            )
            # One dot per gamma combo
            nc = n_combo_cls[cls]
            if nc > 0:
                dot_positions = np.full(nc, means[i, j])
                jitter = np.random.normal(loc=0.0, scale=bar_width/10, size=nc)
                ax.scatter(
                    np.full(dot_positions.shape, x[j] + offset) + jitter,
                    dot_positions,
                    s=4,
                    color="darkgrey",
                    alpha=0.5,
                    marker="."
                )

    # ANOVA stars
    for j, tw in enumerate(time_order):
        group_vals = [pv for pv in pooled_vals[tw] if pv.size > 0]
        if len(group_vals) >= 2:
            _, p = f_oneway(*group_vals)
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            if stars:
                col_heights = means[:, j] + sems[:, j]
                max_bar = np.nanmax(col_heights)
                y_star = max_bar + 0.02
                ax.text(x[j], y_star, stars, ha="center", va="bottom", color="red", fontsize=12)

    # Show n_subj/n_combo per class
    text_lines = []
    for cls in class_labels:
        text_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={n_subj_cls[cls]}, n_combo={n_combo_cls[cls]}")
    ax.text(0.02, 0.95, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize="6")

    ax.set_xticks(x)
    ax.set_xticklabels(["Enc", "D1", "D2", "D3"], rotation=0)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(title="Class", loc="upper right", fontsize="6")
    ax.set_title("Gamma-Only Channels", fontsize=10)

def plot_panel_mr_enc(ax, tw_dict, subj_enc_dict, combo_enc_dict):
    """
    Panel 3: MiniRocket Encoding-Trained.
      - subj_enc_dict:  { cls -> set(subjects) }
      - combo_enc_dict: { cls -> set(combo_indices) }
    """
    time_order   = TIME_WINDOWS
    n_windows    = len(time_order)
    class_labels = [1,2,3,4]
    bar_width    = 0.2
    x            = np.arange(n_windows)

    # Compute means and SEMs
    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            vals = np.array(tw_dict[tw][cls], dtype=float)
            if vals.size > 0:
                means[i, j] = np.nanmean(vals)
                sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
                pooled_vals[tw].append(vals)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
                pooled_vals[tw].append(np.array([], dtype=float))

    # n_subj & n_combo per class
    n_subj_cls  = { cls: len(subj_enc_dict[cls])   for cls in class_labels }
    n_combo_cls = { cls: len(combo_enc_dict[cls])  for cls in class_labels }

    # Plot bars + points
    for i, cls in enumerate(class_labels):
        offset = (i - 1.5) * bar_width
        for j, tw in enumerate(time_order):
            ax.bar(
                x[j] + offset,
                means[i, j],
                width=bar_width,
                yerr=sems[i, j],
                label=CLASS_LABEL_MAP[cls] if j == 0 else "_nolegend_",
                capsize=4
            )
            nc = n_combo_cls[cls]
            if nc > 0:
                dot_positions = np.full(nc, means[i, j])
                jitter = np.random.normal(loc=0.0, scale=bar_width/10, size=nc)
                ax.scatter(
                    np.full(dot_positions.shape, x[j] + offset) + jitter,
                    dot_positions,
                    s=4,
                    color="darkgrey",
                    alpha=0.5,
                    marker="."
                )

    # ANOVA stars
    for j, tw in enumerate(time_order):
        group_vals = [pv for pv in pooled_vals[tw] if pv.size > 0]
        if len(group_vals) >= 2:
            _, p = f_oneway(*group_vals)
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            if stars:
                col_heights = means[:, j] + sems[:, j]
                max_bar = np.nanmax(col_heights)
                y_star = max_bar + 0.02
                ax.text(x[j], y_star, stars, ha="center", va="bottom", color="red", fontsize=12)

    # Show n_subj/n_combo per class
    text_lines = []
    for cls in class_labels:
        text_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={n_subj_cls[cls]}, n_combo={n_combo_cls[cls]}")
    ax.text(0.02, 0.95, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize="6")

    ax.set_xticks(x)
    ax.set_xticklabels(["Enc", "D1", "D2", "D3"], rotation=0)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(title="Class", loc="upper right", fontsize="6")
    ax.set_title("MR Encoding-Trained", fontsize=10)

def plot_panel_mr_dp1(ax, tw_dict, subj_dp1_dict, combo_dp1_dict):
    """
    Panel 4: MiniRocket DP1-Trained.
      - subj_dp1_dict:  { cls -> set(subjects) }
      - combo_dp1_dict: { cls -> set(combo_indices) }
    """
    time_order   = TIME_WINDOWS
    n_windows    = len(time_order)
    class_labels = [1,2,3,4]
    bar_width    = 0.2
    x            = np.arange(n_windows)

    # Compute means and SEMs
    means = np.zeros((4, n_windows))
    sems  = np.zeros((4, n_windows))
    pooled_vals = { tw: [] for tw in time_order }

    for i, cls in enumerate(class_labels):
        for j, tw in enumerate(time_order):
            vals = np.array(tw_dict[tw][cls], dtype=float)
            if vals.size > 0:
                means[i, j] = np.nanmean(vals)
                sems[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
                pooled_vals[tw].append(vals)
            else:
                means[i, j] = np.nan
                sems[i, j]  = np.nan
                pooled_vals[tw].append(np.array([], dtype=float))

    # n_subj & n_combo per class
    n_subj_cls  = { cls: len(subj_dp1_dict[cls])   for cls in class_labels }
    n_combo_cls = { cls: len(combo_dp1_dict[cls])  for cls in class_labels }

    # Plot bars + points
    for i, cls in enumerate(class_labels):
        offset = (i - 1.5) * bar_width
        for j, tw in enumerate(time_order):
            ax.bar(
                x[j] + offset,
                means[i, j],
                width=bar_width,
                yerr=sems[i, j],
                label=CLASS_LABEL_MAP[cls] if j == 0 else "_nolegend_",
                capsize=4
            )
            nc = n_combo_cls[cls]
            if nc > 0:
                dot_positions = np.full(nc, means[i, j])
                jitter = np.random.normal(loc=0.0, scale=bar_width/10, size=nc)
                ax.scatter(
                    np.full(dot_positions.shape, x[j] + offset) + jitter,
                    dot_positions,
                    s=4,
                    color="darkgrey",
                    alpha=0.5,
                    marker="."
                )

    # ANOVA stars
    for j, tw in enumerate(time_order):
        group_vals = [pv for pv in pooled_vals[tw] if pv.size > 0]
        if len(group_vals) >= 2:
            _, p = f_oneway(*group_vals)
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            if stars:
                col_heights = means[:, j] + sems[:, j]
                max_bar = np.nanmax(col_heights)
                y_star = max_bar + 0.02
                ax.text(x[j], y_star, stars, ha="center", va="bottom", color="red", fontsize=12)

    # Show n_subj/n_combo per class
    text_lines = []
    for cls in class_labels:
        text_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={n_subj_cls[cls]}, n_combo={n_combo_cls[cls]}")
    ax.text(0.02, 0.95, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize="6")

    ax.set_xticks(x)
    ax.set_xticklabels(["Enc", "D1", "D2", "D3"], rotation=0)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(title="Class", loc="upper right", fontsize="6")
    ax.set_title("MR DP1-Trained", fontsize=10)


# ───────────────────────────────────────────────────────────────────────────────
#  MAIN SCRIPT
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sig_folder   = os.path.join(OUTPUT_DIR, "significant")
    insig_folder = os.path.join(OUTPUT_DIR, "insignificant")
    os.makedirs(sig_folder, exist_ok=True)
    os.makedirs(insig_folder, exist_ok=True)

    # 1) Read region counts (CSV has header: Region, SubjectCount)
    df_regions = pd.read_csv(REGION_COUNTS_CSV)
    df_regions["Region"] = df_regions["Region"].str.strip("[]'")
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

    # 4) Load MiniRocket filter dicts
    mr_enc_dict = read_minirocket_csv(MR_ENCODING_CSV, "D1_signif_vsFix")
    mr_dp1_dict = read_minirocket_csv(MR_DP1_CSV,       "D2_signif_vsFix")

    # 5) Prepare nested structures for four categories:
    #    Each maps (r1,r2) → { fb → { tw → { cls → list(vals) } } }.
    data_all    = { (r1, r2): { fb: { tw: {1:[],2:[],3:[],4:[]} for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }
    subj_all    = { (r1, r2): { fb: { tw: set() for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }
    combo_all   = { (r1, r2): { fb: { tw: set() for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }

    data_gamma  = { (r1, r2): { fb: { tw: {1:[],2:[],3:[],4:[]} for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }
    subj_gamma  = { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }
    combo_gamma = { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }

    data_mr_enc = { (r1, r2): { fb: { tw: {1:[],2:[],3:[],4:[]} for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }
    subj_enc_mr = { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }
    combo_enc_mr= { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }

    data_mr_dp1 = { (r1, r2): { fb: { tw: {1:[],2:[],3:[],4:[]} for tw in TIME_WINDOWS } for fb in FREQ_BANDS } for (r1,r2) in roi_pairs }
    subj_dp1_mr = { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }
    combo_dp1_mr= { (r1, r2): { cls: set() for cls in [1,2,3,4] } for (r1,r2) in roi_pairs }

    # 6) Loop over all subject-session PPL .mat files
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]
    print(f"Found {len(mat_files)} subject-session files.\n")

    for fname in mat_files:
        subj = fname.split("_")[0]
        session = fname.replace(f"{subj}_", "").replace(FILE_SUFFIX, "")
        mat_path = os.path.join(DATA_DIR, fname)
        print(f"Loading file: {fname}")

        combo_labels, trialinfo, PPL_data, sigchans1, sigchans2, chancombo, bipolarAnat_orig = load_subject_data(mat_path)

        # 6a) Identify valid trials (drop any trial with NaN in trialinfo)
        valid_trials_mask = ~np.isnan(trialinfo).any(axis=1)
        nTrials = trialinfo.shape[0]
        nValid  = valid_trials_mask.sum()
        print(f"  Total trials: {nTrials}, Valid trials: {nValid}")

        if nValid == 0:
            print("  → No valid trials, skipping this file.\n")
            continue

        trialinfo_valid = trialinfo[valid_trials_mask, :]
        trial_classes   = trialinfo_valid[:, 6].astype(int)

        PPL_fix_valid = PPL_data["Fixation"][valid_trials_mask, :, :]
        PPL_enc_valid = PPL_data["Encoding"][valid_trials_mask, :, :]
        PPL_dp1_valid = PPL_data["DP1"][valid_trials_mask, :, :]
        PPL_dp2_valid = PPL_data["DP2"][valid_trials_mask, :, :]
        PPL_dp3_valid = PPL_data["DP3"][valid_trials_mask, :, :]

        nValid, nCombos, nFreqs = PPL_fix_valid.shape

        # 6b) Compute dPPL
        dPPL = {
            "Encoding": PPL_enc_valid - PPL_fix_valid,
            "DP1":      PPL_dp1_valid - PPL_fix_valid,
            "DP2":      PPL_dp2_valid - PPL_fix_valid,
            "DP3":      PPL_dp3_valid - PPL_fix_valid
        }

        # 6c) Loop over each ROI-pair
        for (r1, r2) in roi_pairs:
            concat1 = r1 + r2
            concat2 = r2 + r1
            combo_idxs = [
                i for i in range(nCombos)
                if (str(combo_labels[i][2]) == concat1) or (str(combo_labels[i][2]) == concat2)
            ]
            if not combo_idxs:
                print(f"    ROI-pair ({r1} vs {r2}): no matching combos, skipping.")
                continue

            combo_idxs = np.array(combo_idxs, dtype=int)
            print(f"    ROI-pair ({r1} vs {r2}): {len(combo_idxs)} combos.")

            # Precompute freq-bin indices for this subject
            band_indices = {
                fb: freq_bin_indices(freqs, FREQ_BANDS[fb])
                for fb in FREQ_BANDS.keys()
            }

            # 6d) For each time window and frequency band, pool values
            for fb, fidx in band_indices.items():
                for tw in TIME_WINDOWS:
                    with np.errstate(invalid="ignore"):
                        arr_band = np.nanmean(dPPL[tw][:, :, fidx], axis=2)  # shape = (nValid, nCombos)

                    for cls in [1, 2, 3, 4]:
                        trial_idxs = np.where(trial_classes == cls)[0]
                        if trial_idxs.size == 0:
                            continue

                        # 6d(i) → All Channels
                        submat_all = arr_band[np.ix_(trial_idxs, combo_idxs)]
                        vals_all   = submat_all.flatten()
                        vals_all   = vals_all[~np.isnan(vals_all)]
                        if vals_all.size > 0:
                            data_all[(r1, r2)][fb][tw][cls].extend(vals_all.tolist())
                            subj_all[(r1, r2)][fb][tw].add(subj)
                            combo_all[(r1, r2)][fb][tw].update(combo_idxs.tolist())

                        # Determine original channel indices for each combo index:
                        #    For combo index i: the two bipolar rows are chancombo[i][0]-1, chancombo[i][1]-1 (MATLAB→0-based).
                        #    bipolarAnat_orig[...] gives the original channel number (1-based).
                        # Build a set of original-channel pairs for those combo_idxs
                        orig_pairs = [
                            (
                                bipolarAnat_orig[int(chancombo[i][0] - 1)],
                                bipolarAnat_orig[int(chancombo[i][1] - 1)]
                            )
                            for i in combo_idxs
                        ]

                        # 6d(ii) → Gamma-only
                        gamma_chans_cls = set(sigchans1[cls - 1].tolist()) | set(sigchans2[cls - 1].tolist())
                        valid_gamma_combos = [
                            combo_idxs[k]
                            for k, (c1, c2) in enumerate(orig_pairs)
                            if (c1 in gamma_chans_cls) and (c2 in gamma_chans_cls)
                        ]
                        if valid_gamma_combos:
                            submat_gam = arr_band[np.ix_(trial_idxs, valid_gamma_combos)]
                            vals_gam   = submat_gam.flatten()
                            vals_gam   = vals_gam[~np.isnan(vals_gam)]
                            if vals_gam.size > 0:
                                data_gamma[(r1, r2)][fb][tw][cls].extend(vals_gam.tolist())
                                subj_gamma[(r1, r2)][cls].add(subj)
                                combo_gamma[(r1, r2)][cls].update(valid_gamma_combos)

                        # 6d(iii) → MR Encoding-trained
                        key_enc = (subj, cls)
                        if key_enc in mr_enc_dict:
                            mr_enc_chans = mr_enc_dict[key_enc]
                            valid_enc_combos = [
                                combo_idxs[k]
                                for k, (c1, c2) in enumerate(orig_pairs)
                                if (c1 in mr_enc_chans) and (c2 in mr_enc_chans)
                            ]
                            if valid_enc_combos:
                                submat_enc = arr_band[np.ix_(trial_idxs, valid_enc_combos)]
                                vals_enc   = submat_enc.flatten()
                                vals_enc   = vals_enc[~np.isnan(vals_enc)]
                                if vals_enc.size > 0:
                                    data_mr_enc[(r1, r2)][fb][tw][cls].extend(vals_enc.tolist())
                                    subj_enc_mr[(r1, r2)][cls].add(subj)
                                    combo_enc_mr[(r1, r2)][cls].update(valid_enc_combos)

                        # 6d(iv) → MR DP1-trained
                        key_dp1 = (subj, cls)
                        if key_dp1 in mr_dp1_dict:
                            mr_dp1_chans = mr_dp1_dict[key_dp1]
                            valid_dp1_combos = [
                                combo_idxs[k]
                                for k, (c1, c2) in enumerate(orig_pairs)
                                if (c1 in mr_dp1_chans) and (c2 in mr_dp1_chans)
                            ]
                            if valid_dp1_combos:
                                submat_dp1 = arr_band[np.ix_(trial_idxs, valid_dp1_combos)]
                                vals_dp1   = submat_dp1.flatten()
                                vals_dp1   = vals_dp1[~np.isnan(vals_dp1)]
                                if vals_dp1.size > 0:
                                    data_mr_dp1[(r1, r2)][fb][tw][cls].extend(vals_dp1.tolist())
                                    subj_dp1_mr[(r1, r2)][cls].add(subj)
                                    combo_dp1_mr[(r1, r2)][cls].update(valid_dp1_combos)

        print("")  # blank line between files

    # 7) Plotting & ANOVA
    total_plots = 0
    for (r1, r2), band_dict in data_all.items():
        for fb, tw_dict_all in band_dict.items():
            # Check if any panel is “significant”
            has_any_all   = check_any_significant(tw_dict_all)
            tw_dict_gamma= data_gamma[(r1, r2)][fb]
            has_any_gamma= check_any_significant(tw_dict_gamma)
            tw_dict_enc  = data_mr_enc[(r1, r2)][fb]
            has_any_enc  = check_any_significant(tw_dict_enc)
            tw_dict_dp1  = data_mr_dp1[(r1, r2)][fb]
            has_any_dp1  = check_any_significant(tw_dict_dp1)

            if not (has_any_all or has_any_gamma or has_any_enc or has_any_dp1):
                print(f"Skipping {r1} vs {r2} | {fb}: no significant data in any panel.")
                continue

            total_plots += 1
            print(f"Plotting {total_plots}: ROI pair {r1} vs {r2}, Frequency band: {fb}")

            fig, axes = plt.subplots(1, 4, figsize=(28, 6), sharey=True)
            plt.suptitle(f"{r1} vs {r2}  |  {fb} band", fontsize=18)

            # PANEL 1: All Channels
            plot_panel_all(
                axes[0],
                tw_dict_all,
                subj_all[(r1, r2)][fb],
                combo_all[(r1, r2)][fb]
            )

            # PANEL 2: Gamma-Only Channels
            plot_panel_gamma(
                axes[1],
                tw_dict_gamma,
                subj_gamma[(r1, r2)],
                combo_gamma[(r1, r2)],
                bipolarAnat_orig
            )

            # PANEL 3: MR Encoding-Trained
            plot_panel_mr_enc(
                axes[2],
                tw_dict_enc,
                subj_enc_mr[(r1, r2)],
                combo_enc_mr[(r1, r2)]
            )

            # PANEL 4: MR DP1-Trained
            plot_panel_mr_dp1(
                axes[3],
                tw_dict_dp1,
                subj_dp1_mr[(r1, r2)],
                combo_dp1_mr[(r1, r2)]
            )

            # Save figure in the “significant” folder
            save_path = os.path.join(
                sig_folder,
                f"{fb}{r1.replace('/', '_')}_{r2.replace('/', '_')}.png"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path, dpi=150)
            plt.close()

    print(f"\nFinished plotting {total_plots} figures. Check '{OUTPUT_DIR}' for output.")
