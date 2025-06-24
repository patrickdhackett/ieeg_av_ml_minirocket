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
MINIROCKET_CSV       = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\13_permutationshuffledtrain_iti\Encoding train\13_onevsall_with_permutationontrain_iti.csv"

# Frequency band definitions (Hz)
FREQ_BANDS = {
    "theta":     (4,   8),
    "beta":      (13,  25),
    "low_gamma": (26,  70),
    "high_gamma":(71, 140),
}

# Time windows—keys must match keys in the PPL_data dict
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
    Given a sorted 1D array freqs and a tuple (low, high),
    return the indices of freqs within [low, high].
    """
    low, high = band
    return np.where((freqs >= low) & (freqs <= high))[0]


def load_subject_data(mat_path):
    """
    Load one subject-session .mat that contains:
      - combolabels   (nCombos x 3): [ROI1, ROI2, concatenatedROI]
      - trialinfo     (nTrials x 8)
      - PPL arrays:   "PPL_fix_all", "PPL_enc_first1k", "PPL_dp_first1k",
                      "PPL_dp_second1k", "PPL_dp_third1k"
      - chancombo     (nCombos x 2): bipolar channel indices per combo
      - sigchans1, sigchans2 (each length-4 cell) listing gamma‐modulated channels per class
    Returns:
      - combo_labels: numpy array (nCombos, 3) of dtype=object
      - trialinfo:    numpy array (nTrials, 8)
      - PPL_data:     dict with keys "Fixation","Encoding","DP1","DP2","DP3",
                      each shape (nTrials, nCombos, nFreqs)
      - chancombo:    numpy array (nCombos, 2)
      - sigchans1:    list of length 4, each an array of bipolar channel indices
      - sigchans2:    list of length 4, each an array of bipolar channel indices
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    combo_labels = mat["combolabels"]      # (nCombos, 3) cell array
    trialinfo    = mat["trialinfo"]        # (nTrials, 8) double
    chancombo    = mat["chancombo"]        # (nCombos, 2) double

    PPL_data = {
        "Fixation": mat["PPL_fix_all"],       # shape: (nTrials, nCombos, nFreqs)
        "Encoding": mat["PPL_enc_first1k"],
        "DP1":      mat["PPL_dp_first1k"],
        "DP2":      mat["PPL_dp_second1k"],
        "DP3":      mat["PPL_dp_third1k"]
    }

    sigchans1 = mat.get("sigchans1", None)  # length-4, each element is an array of ints
    sigchans2 = mat.get("sigchans2", None)

    return combo_labels, trialinfo, PPL_data, chancombo, sigchans1, sigchans2


# ───────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Read region counts (CSV has header: Region, SubjectCount)
    df_regions = pd.read_csv(REGION_COUNTS_CSV)
    # Convert "['L ACC']" → "L ACC"
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

    # 4) Read minirocket CSV once; filter for D1_signif_vsFix == True
    df_mr = pd.read_csv(MINIROCKET_CSV)
    # We only need Subject, ChannelIndex, ClassLabel, D1_signif_vsFix
    df_mr = df_mr[["Subject", "ChannelIndex", "ClassLabel", "D1_signif_vsFix"]]

    # 5) Prepare nested structures for data accumulation:
    #
    #    data_all[(r1,r2)][fb][tw][cls] = list of ΔPPL floats (all channels)
    #    data_gamma[(r1,r2)][fb][tw][cls] = list of ΔPPL floats (gamma‐only)
    #    data_mr[(r1,r2)][fb][tw][cls]   = list of ΔPPL floats (minirocket‐only)
    #
    #    subj_all[(r1,r2)][fb][tw]       = set of subjects contributing all‐channel
    #    combo_all[(r1,r2)][fb][tw]      = set of combos contributing all‐channel
    #
    #    subj_gamma_perclass[(r1,r2)][cls]  = set of subjects contributing gamma‐only for class
    #    combo_gamma_perclass[(r1,r2)][cls] = set of combo indices gamma‐only for class
    #
    #    subj_mr_perclass[(r1,r2)][cls]  = set of subjects contributing MR‐only for class
    #    combo_mr_perclass[(r1,r2)][cls] = set of combo indices MR‐only for class
    #
    data_all              = {}
    data_gamma            = {}
    data_mr               = {}
    subj_all              = {}
    combo_all             = {}
    subj_gamma_perclass   = {}
    combo_gamma_perclass  = {}
    subj_mr_perclass      = {}
    combo_mr_perclass     = {}

    for (r1, r2) in roi_pairs:
        data_all[(r1, r2)]             = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        data_gamma[(r1, r2)]           = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        data_mr[(r1, r2)]              = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_all[(r1, r2)]             = {
            fb: { tw: set() for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        combo_all[(r1, r2)]            = {
            fb: { tw: set() for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS
        }
        subj_gamma_perclass[(r1, r2)]  = { cls: set() for cls in [1, 2, 3, 4] }
        combo_gamma_perclass[(r1, r2)] = { cls: set() for cls in [1, 2, 3, 4] }
        subj_mr_perclass[(r1, r2)]     = { cls: set() for cls in [1, 2, 3, 4] }
        combo_mr_perclass[(r1, r2)]    = { cls: set() for cls in [1, 2, 3, 4] }

    # 6) Loop over all subject-session PPL files
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]
    print(f"Found {len(mat_files)} subject-session files.\n")

    for fname in mat_files:
        subj = fname.split("_")[0]
        session = fname.replace(f"{subj}_", "").replace(FILE_SUFFIX, "")
        mat_path = os.path.join(DATA_DIR, fname)
        print(f"Loading: {fname}")

        (combo_labels,
         trialinfo,
         PPL_data,
         chancombo,
         sigchans1,
         sigchans2) = load_subject_data(mat_path)

        # 6a) Drop any trial that has a NaN anywhere in trialinfo
        valid_trials_mask = ~np.isnan(trialinfo).any(axis=1)
        nTrials = trialinfo.shape[0]
        nValid = valid_trials_mask.sum()
        print(f"  Total trials: {nTrials}, Valid trials: {nValid}")

        if nValid == 0:
            print("  → No valid trials, skipping entire file.\n")
            continue

        trialinfo_valid = trialinfo[valid_trials_mask, :]
        trial_classes   = trialinfo_valid[:, 6].astype(int)  # class ∈ {1,2,3,4}

        # Filter PPL arrays by valid trials
        PPL_fix_valid = PPL_data["Fixation"][valid_trials_mask, :, :]
        PPL_enc_valid = PPL_data["Encoding"][valid_trials_mask, :, :]
        PPL_dp1_valid = PPL_data["DP1"][valid_trials_mask, :, :]
        PPL_dp2_valid = PPL_data["DP2"][valid_trials_mask, :, :]
        PPL_dp3_valid = PPL_data["DP3"][valid_trials_mask, :, :]

        nValid, nCombos, nFreqs = PPL_fix_valid.shape

        # Compute ΔPPL = (window) − (fixation), for each trial/combination/freq
        dPPL = {
            "Encoding": PPL_enc_valid - PPL_fix_valid,
            "DP1":      PPL_dp1_valid - PPL_fix_valid,
            "DP2":      PPL_dp2_valid - PPL_fix_valid,
            "DP3":      PPL_dp3_valid - PPL_fix_valid
        }

        # Build gamma‐channel sets per class
        gamma_sets = {}
        if sigchans1 is not None and sigchans2 is not None:
            for cls in [1, 2, 3, 4]:
                arr1 = np.atleast_1d(sigchans1[cls-1]).astype(int).ravel()
                arr2 = np.atleast_1d(sigchans2[cls-1]).astype(int).ravel()
                gamma_sets[cls] = set(arr1.tolist() + arr2.tolist())
        else:
            for cls in [1, 2, 3, 4]:
                gamma_sets[cls] = set()

        # Build minirocket‐significant channel sets per class for this subject
        sub_df = df_mr[df_mr["Subject"] == subj]
        mr_sets = {}
        for cls in [1, 2, 3, 4]:
            cls_df = sub_df[(sub_df["ClassLabel"] == cls) & (sub_df["D1_signif_vsFix"] == True)]
            mr_sets[cls] = set(cls_df["ChannelIndex"].astype(int).tolist())

        # 6b) For each ROI‐pair (r1, r2), find matching combo indices
        for (r1, r2) in roi_pairs:
            concat1 = r1 + r2
            concat2 = r2 + r1
            combo_idxs = [
                i for i in range(nCombos)
                if (str(combo_labels[i][2]) == concat1) or (str(combo_labels[i][2]) == concat2)
            ]
            if not combo_idxs:
                print(f"    ROI-pair ({r1} vs {r2}): 0 combos, skipping.")
                continue

            combo_idxs = np.array(combo_idxs, dtype=int)
            print(f"    ROI-pair ({r1} vs {r2}): {len(combo_idxs)} combos found.")

            # Precompute freq‐bin indices for each band
            band_indices = { fb: freq_bin_indices(freqs, FREQ_BANDS[fb]) for fb in FREQ_BANDS }

            # 6c) For each time window and freq band, pool ΔPPL per class
            for fb, fidx in band_indices.items():
                if fidx.size == 0:
                    print(f"      Band {fb}: no freq bins, skipping.")
                    continue

                for tw in TIME_WINDOWS:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        arr_band = np.nanmean(dPPL[tw][:, :, fidx], axis=2)
                        # arr_band.shape == (nValidTrials, nCombos)

                    for cls in [1, 2, 3, 4]:
                        trial_idxs = np.where(trial_classes == cls)[0]
                        if trial_idxs.size == 0:
                            print(f"        Class {cls}: no valid trials, skipping.")
                            continue

                        # --- ALL CHANNELS (per‐combo average across trials) ---
                        submat_all = arr_band[np.ix_(trial_idxs, combo_idxs)]  # (nTrials_cls, nCombos_pair)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_per_combo = np.nanmean(submat_all, axis=0)        # (nCombos_pair,)
                        vals_all = mean_per_combo[~np.isnan(mean_per_combo)]
                        if vals_all.size > 0:
                            data_all[(r1, r2)][fb][tw][cls].extend(vals_all.tolist())
                            subj_all[(r1, r2)][fb][tw].add(subj)
                            combo_all[(r1, r2)][fb][tw].update(combo_idxs.tolist())

                        # --- GAMMA‐ONLY CHANNELS ---
                        gamma_set = gamma_sets[cls]
                        valid_gamma_combos = [
                            idx for idx in combo_idxs
                            if (int(chancombo[idx, 0]) in gamma_set) and (int(chancombo[idx, 1]) in gamma_set)
                        ]
                        if valid_gamma_combos:
                            valid_gamma_combos = np.array(valid_gamma_combos, dtype=int)
                            submat_gamma = arr_band[np.ix_(trial_idxs, valid_gamma_combos)]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                mean_per_gamma = np.nanmean(submat_gamma, axis=0)
                            vals_g = mean_per_gamma[~np.isnan(mean_per_gamma)]
                            if vals_g.size > 0:
                                data_gamma[(r1, r2)][fb][tw][cls].extend(vals_g.tolist())
                                subj_gamma_perclass[(r1, r2)][cls].add(subj)
                                combo_gamma_perclass[(r1, r2)][cls].update(valid_gamma_combos.tolist())

                        # --- MINIROCKET‐ONLY CHANNELS ---
                        mr_set = mr_sets[cls]
                        valid_mr_combos = [
                            idx for idx in combo_idxs
                            if (int(chancombo[idx, 0]) in mr_set) and (int(chancombo[idx, 1]) in mr_set)
                        ]
                        if valid_mr_combos:
                            valid_mr_combos = np.array(valid_mr_combos, dtype=int)
                            submat_mr = arr_band[np.ix_(trial_idxs, valid_mr_combos)]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                mean_per_mr = np.nanmean(submat_mr, axis=0)
                            vals_m = mean_per_mr[~np.isnan(mean_per_mr)]
                            if vals_m.size > 0:
                                data_mr[(r1, r2)][fb][tw][cls].extend(vals_m.tolist())
                                subj_mr_perclass[(r1, r2)][cls].add(subj)
                                combo_mr_perclass[(r1, r2)][cls].update(valid_mr_combos.tolist())

        print("")  # blank line between files

    # ───────────────────────────────────────────────────────────────────────────────
    #  7) Plotting & ANOVA (All Channels, Gamma‐Only, MiniRocket‐Only side‐by‐side)
    # ───────────────────────────────────────────────────────────────────────────────
    total_plots = 0
    for (r1, r2), _ in data_all.items():
        for fb in FREQ_BANDS.keys():
            tw_dict_all   = data_all[(r1, r2)][fb]
            tw_dict_gamma = data_gamma[(r1, r2)][fb]
            tw_dict_mr    = data_mr[(r1, r2)][fb]

            has_any_all   = any(len(tw_dict_all[tw][cls]) > 0 for tw in TIME_WINDOWS for cls in [1,2,3,4])
            has_any_gamma = any(len(tw_dict_gamma[tw][cls]) > 0 for tw in TIME_WINDOWS for cls in [1,2,3,4])
            has_any_mr    = any(len(tw_dict_mr[tw][cls]) > 0 for tw in TIME_WINDOWS for cls in [1,2,3,4])

            if not (has_any_all or has_any_gamma or has_any_mr):
                print(f"Skipping {r1} vs {r2} | {fb}: no data at all.")
                continue

            total_plots += 1
            print(f"Plotting {total_plots}: ROI pair {r1} vs {r2}, Frequency band: {fb}")

            fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
            plt.suptitle(f"{r1} vs {r2}  |  {fb} band", fontsize=16)

            time_order   = ["Encoding", "DP1", "DP2", "DP3"]
            n_windows    = len(time_order)
            class_labels = [1, 2, 3, 4]
            bar_width    = 0.16
            x            = np.arange(n_windows)

            # --- Plot 1: All Channels ---
            ax_all = axes[0]
            means_all = np.zeros((4, n_windows))
            sems_all  = np.zeros((4, n_windows))
            pooled_vals_all = { tw: [] for tw in time_order }

            for i, cls in enumerate(class_labels):
                for j, tw in enumerate(time_order):
                    vals = np.array(tw_dict_all[tw][cls], dtype=float)
                    if vals.size > 0:
                        means_all[i, j] = np.nanmean(vals)
                        sems_all[i, j]  = np.nanstd(vals, ddof=1) / np.sqrt(vals.size)
                    else:
                        means_all[i, j] = np.nan
                        sems_all[i, j]  = np.nan
                    pooled_vals_all[tw].append(vals)

            for i, cls in enumerate(class_labels):
                offset = (i - 1.5) * bar_width
                bar_positions = x + offset
                ax_all.bar(
                    bar_positions,
                    means_all[i, :],
                    width=bar_width,
                    yerr=sems_all[i, :],
                    label=CLASS_LABEL_MAP[cls],
                    capsize=3
                )
                for j in range(n_windows):
                    vals = np.array(tw_dict_all[time_order[j]][cls], dtype=float)
                    if vals.size == 0:
                        continue
                    jitter = np.random.normal(loc=0.0, scale=(bar_width/6), size=vals.size)
                    ax_all.scatter(
                        np.full(vals.shape, x[j] + offset) + jitter,
                        vals,
                        s=6,
                        marker=".",
                        color="#888888",
                        alpha=0.15
                    )

            max_bar_all = np.nanmax(means_all + sems_all) if np.isfinite(means_all).any() else 0
            top_all = max_bar_all

            nsubj_all_total = len(set().union(
                *[subj_all[(r1, r2)][fb][tw] for tw in TIME_WINDOWS]
            ))
            ncomb_all_total = len(set().union(
                *[combo_all[(r1, r2)][fb][tw] for tw in TIME_WINDOWS]
            ))
            ax_all.text(0.5, top_all + 0.08,
                        f"n_subj={nsubj_all_total}, n_combo={ncomb_all_total}",
                        ha="center", va="bottom", fontsize=10)

            for j, tw in enumerate(time_order):
                group_vals = [pv for pv in pooled_vals_all[tw] if pv.size > 0]
                if len(group_vals) >= 2:
                    F, p = f_oneway(*group_vals)
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    elif p < 0.05:
                        star = "*"
                    else:
                        star = ""
                    if star:
                        ax_all.text(x[j], top_all + 0.12,
                                    star, ha="center", va="bottom",
                                    color="red", fontsize=16)

            ax_all.set_xticks(x)
            ax_all.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
            ax_all.set_xlabel("Time Window")
            ax_all.set_ylabel("DeltaPPL (Window − Fix)")
            ax_all.set_title("All Channels")
            ax_all.legend(title="Class", loc="upper right", fontsize="small")
            ax_all.axhline(0, color="gray", linewidth=1)

            # --- Plot 2: Gamma‐Only Channels ---
            ax_gamma = axes[1]
            means_g  = np.zeros((4, n_windows))
            sems_g   = np.zeros((4, n_windows))
            pooled_vals_g = { tw: [] for tw in time_order }

            for i, cls in enumerate(class_labels):
                for j, tw in enumerate(time_order):
                    vals_g = np.array(tw_dict_gamma[tw][cls], dtype=float)
                    if vals_g.size > 0:
                        means_g[i, j] = np.nanmean(vals_g)
                        sems_g[i, j]  = np.nanstd(vals_g, ddof=1) / np.sqrt(vals_g.size)
                    else:
                        means_g[i, j] = np.nan
                        sems_g[i, j]  = np.nan
                    pooled_vals_g[tw].append(vals_g)

            for i, cls in enumerate(class_labels):
                offset = (i - 1.5) * bar_width
                bar_positions = x + offset
                ax_gamma.bar(
                    bar_positions,
                    means_g[i, :],
                    width=bar_width,
                    yerr=sems_g[i, :],
                    label=CLASS_LABEL_MAP[cls],
                    capsize=3
                )
                for j in range(n_windows):
                    vals_g = np.array(tw_dict_gamma[time_order[j]][cls], dtype=float)
                    if vals_g.size == 0:
                        continue
                    jitter = np.random.normal(loc=0.0, scale=(bar_width/6), size=vals_g.size)
                    ax_gamma.scatter(
                        np.full(vals_g.shape, x[j] + offset) + jitter,
                        vals_g,
                        s=6,
                        marker=".",
                        color="#888888",
                        alpha=0.15
                    )

            max_bar_g = np.nanmax(means_g + sems_g) if np.isfinite(means_g).any() else 0
            top_g = max_bar_g

            # Show n_subj & n_combo per class (gamma only)
            info_lines = []
            for cls in class_labels:
                nsubj_g_cls = len(subj_gamma_perclass[(r1, r2)][cls])
                ncomb_g_cls = len(combo_gamma_perclass[(r1, r2)][cls])
                info_lines.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={nsubj_g_cls}, n_combo={ncomb_g_cls}")
            textstr_g = "\n".join(info_lines)
            ax_gamma.text(0.02, 0.98, textstr_g, transform=ax_gamma.transAxes,
                          fontsize=8, va='top', ha='left')

            for j, tw in enumerate(time_order):
                group_vals_g = [pv for pv in pooled_vals_g[tw] if pv.size > 0]
                if len(group_vals_g) >= 2:
                    F, p = f_oneway(*group_vals_g)
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    elif p < 0.05:
                        star = "*"
                    else:
                        star = ""
                    if star:
                        ax_gamma.text(x[j], top_g + 0.08,
                                      star, ha="center", va="bottom",
                                      color="red", fontsize=16)

            ax_gamma.set_xticks(x)
            ax_gamma.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
            ax_gamma.set_xlabel("Time Window")
            ax_gamma.set_title("Gamma-Only Channels")
            ax_gamma.legend(title="Class", loc="upper right", fontsize="small")
            ax_gamma.axhline(0, color="gray", linewidth=1)

            # --- Plot 3: MiniRocket‐Only Channels ---
            ax_mr = axes[2]
            means_m  = np.zeros((4, n_windows))
            sems_m   = np.zeros((4, n_windows))
            pooled_vals_m = { tw: [] for tw in time_order }

            for i, cls in enumerate(class_labels):
                for j, tw in enumerate(time_order):
                    vals_m = np.array(tw_dict_mr[tw][cls], dtype=float)
                    if vals_m.size > 0:
                        means_m[i, j] = np.nanmean(vals_m)
                        sems_m[i, j]  = np.nanstd(vals_m, ddof=1) / np.sqrt(vals_m.size)
                    else:
                        means_m[i, j] = np.nan
                        sems_m[i, j]  = np.nan
                    pooled_vals_m[tw].append(vals_m)

            for i, cls in enumerate(class_labels):
                offset = (i - 1.5) * bar_width
                bar_positions = x + offset
                ax_mr.bar(
                    bar_positions,
                    means_m[i, :],
                    width=bar_width,
                    yerr=sems_m[i, :],
                    label=CLASS_LABEL_MAP[cls],
                    capsize=3
                )
                for j in range(n_windows):
                    vals_m = np.array(tw_dict_mr[time_order[j]][cls], dtype=float)
                    if vals_m.size == 0:
                        continue
                    jitter = np.random.normal(loc=0.0, scale=(bar_width/6), size=vals_m.size)
                    ax_mr.scatter(
                        np.full(vals_m.shape, x[j] + offset) + jitter,
                        vals_m,
                        s=6,
                        marker=".",
                        color="#888888",
                        alpha=0.15
                    )

            max_bar_m = np.nanmax(means_m + sems_m) if np.isfinite(means_m).any() else 0
            top_m = max_bar_m

            # Show n_subj & n_combo per class (MR only)
            info_lines_m = []
            for cls in class_labels:
                nsubj_m_cls = len(subj_mr_perclass[(r1, r2)][cls])
                ncomb_m_cls = len(combo_mr_perclass[(r1, r2)][cls])
                info_lines_m.append(f"{CLASS_LABEL_MAP[cls]}: n_subj={nsubj_m_cls}, n_combo={ncomb_m_cls}")
            textstr_m = "\n".join(info_lines_m)
            ax_mr.text(0.02, 0.98, textstr_m, transform=ax_mr.transAxes,
                       fontsize=8, va='top', ha='left')

            for j, tw in enumerate(time_order):
                group_vals_m = [pv for pv in pooled_vals_m[tw] if pv.size > 0]
                if len(group_vals_m) >= 2:
                    F, p = f_oneway(*group_vals_m)
                    if p < 0.001:
                        star = "***"
                    elif p < 0.01:
                        star = "**"
                    elif p < 0.05:
                        star = "*"
                    else:
                        star = ""
                    if star:
                        ax_mr.text(x[j], top_m + 0.08,
                                   star, ha="center", va="bottom",
                                   color="red", fontsize=16)

            ax_mr.set_xticks(x)
            ax_mr.set_xticklabels(["Encoding", "Delay1", "Delay2", "Delay3"])
            ax_mr.set_xlabel("Time Window")
            ax_mr.set_title("MiniRocket-Only Channels")
            ax_mr.legend(title="Class", loc="upper right", fontsize="small")
            ax_mr.axhline(0, color="gray", linewidth=1)

            # Adjust layout and save
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # leave room for suptitle
            fname_combined = f"{r1.replace('/', '_')}_{r2.replace('/', '_')}_{fb}_combined.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname_combined), dpi=150)
            plt.close()

    print(f"\nFinished plotting {total_plots} combined figures. Check '{OUTPUT_DIR}' for output.")
