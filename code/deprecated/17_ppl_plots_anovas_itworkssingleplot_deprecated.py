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
OUTPUT_DIR        = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\17_plots_ppl"

# Frequency band definitions
FREQ_BANDS = {
    "theta":     (4,   8),
    "beta":      (13,  25),
    "low_gamma": (26,  70),
    "high_gamma":(71, 140),
}

# Time windows—keys must match PPL_data keys
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
      - combolabels (nCombos x 3 cell): [ROI1, ROI2, concatenatedROI]
      - trialinfo   (nTrials x 8 array)
      - PPL arrays: "PPL_fix_all", "PPL_enc_first1k", "PPL_dp_first1k",
                    "PPL_dp_second1k", "PPL_dp_third1k"
    Returns:
      - combo_labels: numpy array (nCombos, 3)
      - trialinfo:    numpy array (nTrials, 8)
      - PPL_data:     dict with keys "Fixation","Encoding","DP1","DP2","DP3",
                      each shape (nTrials, nCombos, nFreqs)
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    combo_labels = mat["combolabels"]
    trialinfo    = mat["trialinfo"]

    PPL_data = {
        "Fixation": mat["PPL_fix_all"],          # shape (nTrials, nCombos, nFreqs)
        "Encoding": mat["PPL_enc_first1k"],       # same shape
        "DP1":      mat["PPL_dp_first1k"],
        "DP2":      mat["PPL_dp_second1k"],
        "DP3":      mat["PPL_dp_third1k"]
    }

    return combo_labels, trialinfo, PPL_data

# ───────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Read region counts (CSV has header: Region, SubjectCount)
    df_regions = pd.read_csv(REGION_COUNTS_CSV)
    # Strip off surrounding brackets/quotes: "['L ACC']" → "L ACC"
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

    # 4) Prepare a nested structure to accumulate dPPL values:
    #    data[(roi1, roi2)][freq_band][time_window][class_label] = list of dPPL floats
    data = {}
    for (r1, r2) in roi_pairs:
        data[(r1, r2)] = {
            fb: { tw: {1: [], 2: [], 3: [], 4: []} for tw in TIME_WINDOWS }
            for fb in FREQ_BANDS.keys()
        }

    # 5) Loop over all subject-session PPL files
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]
    print(f"Found {len(mat_files)} subject-session files.\n")

    for fname in mat_files:
        subj = fname.split("_")[0]
        session = fname.replace(f"{subj}_", "").replace(FILE_SUFFIX, "")
        mat_path = os.path.join(DATA_DIR, fname)
        print(f"Loading file: {fname}")

        combo_labels, trialinfo, PPL_data = load_subject_data(mat_path)

        # 5a) Identify valid trials: drop any trial with NaN anywhere in trialinfo
        valid_trials_mask = ~np.isnan(trialinfo).any(axis=1)
        nTrials = trialinfo.shape[0]
        nValid = valid_trials_mask.sum()
        print(f"  Total trials: {nTrials}, Valid trials: {nValid}")

        if nValid == 0:
            print("  → No valid trials, skipping this file.\n")
            continue

        # Filter trialinfo and PPL_data by valid trials
        trialinfo_valid = trialinfo[valid_trials_mask, :]
        trial_classes   = trialinfo_valid[:, 6].astype(int)  # safe to cast now

        PPL_fix_valid = PPL_data["Fixation"][valid_trials_mask, :, :]
        PPL_enc_valid = PPL_data["Encoding"][valid_trials_mask, :, :]
        PPL_dp1_valid = PPL_data["DP1"][valid_trials_mask, :, :]
        PPL_dp2_valid = PPL_data["DP2"][valid_trials_mask, :, :]
        PPL_dp3_valid = PPL_data["DP3"][valid_trials_mask, :, :]

        nValid, nCombos, nFreqs = PPL_fix_valid.shape

        # Compute dPPL for each window = PPL_window_valid - PPL_fix_valid
        dPPL = {
            "Encoding": PPL_enc_valid - PPL_fix_valid,
            "DP1":      PPL_dp1_valid - PPL_fix_valid,
            "DP2":      PPL_dp2_valid - PPL_fix_valid,
            "DP3":      PPL_dp3_valid - PPL_fix_valid
        }

        # 5b) For each ROI-pair, find corresponding combo indices
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

            # 5c) For each time window and frequency band, pool dPPL per class
            for fb, fidx in band_indices.items():
                for tw in TIME_WINDOWS:
                    # Use nanmean over the frequency dimension to ignore NaNs there
                    with np.errstate(invalid="ignore"):  # suppress "mean of empty slice" warnings
                        arr_band = np.nanmean(dPPL[tw][:, :, fidx], axis=2)  # shape = (nValid, nCombos)
                    for cls in [1, 2, 3, 4]:
                        trial_idxs = np.where(trial_classes == cls)[0]
                        if trial_idxs.size == 0:
                            print(f"      Class {cls}: no valid trials, skipping.")
                            continue

                        submat = arr_band[np.ix_(trial_idxs, combo_idxs)]  
                        vals = submat.flatten()
                        # Remove any NaNs that might remain 
                        vals = vals[~np.isnan(vals)]
                        if vals.size > 0:
                            data[(r1, r2)][fb][tw][cls].extend(vals.tolist())

        print("")  # blank line between files

    # 6) Plotting & ANOVA
    total_plots = 0
    for (r1, r2), band_dict in data.items():
        for fb, tw_dict in band_dict.items():
            # Check whether any class/time window has data
            has_data = False
            for tw in TIME_WINDOWS:
                for cls in [1, 2, 3, 4]:
                    if len(tw_dict[tw][cls]) > 0:
                        has_data = True
                        break
                if has_data:
                    break

            if not has_data:
                print(f"Skipping plot for ({r1} vs {r2}, {fb}): no data.")
                continue

            total_plots += 1
            print(f"Plotting {total_plots}: ROI pair {r1} vs {r2}, Frequency band: {fb}")

            time_order   = ["Encoding", "DP1", "DP2", "DP3"]
            n_windows    = len(time_order)
            class_labels = [1, 2, 3, 4]
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
                    else:
                        means[i, j] = np.nan
                        sems[i, j]  = np.nan
                    pooled_vals[tw].append(vals)

            plt.figure(figsize=(10, 6))

            # Plot bars + individual points
            for i, cls in enumerate(class_labels):
                offset = (i - 1.5) * bar_width
                bar_positions = x + offset
                plt.bar(
                    bar_positions,
                    means[i, :],
                    width=bar_width,
                    yerr=sems[i, :],
                    label=CLASS_LABEL_MAP[cls],
                    capsize=4
                )
                for j in range(n_windows):
                    vals = np.array(tw_dict[time_order[j]][cls], dtype=float)
                    if vals.size == 0:
                        continue
                    jitter = np.random.normal(loc=0.0, scale=bar_width/6, size=vals.size)
                    plt.scatter(
                        np.full(vals.shape, x[j] + offset) + jitter,
                        vals,
                        s=10,
                        color="black",
                        alpha=0.6
                    )

            # Run one-way ANOVA at each time window; if p < 0.05, mark '*'
            for j, tw in enumerate(time_order):
                group_vals = [pv for pv in pooled_vals[tw] if pv.size > 0]
                if len(group_vals) >= 2:
                    F, p = f_oneway(*group_vals)
                    if p < 0.05:
                        col_heights = means[:, j] + sems[:, j]
                        max_bar = np.nanmax(col_heights)
                        y_star = max_bar + 0.05
                        plt.text(x[j], y_star, "*", ha="center", va="bottom", color="red", fontsize=16)

            plt.xticks(x, ["Encoding", "Delay1", "Delay2", "Delay3"])
            plt.xlabel("Time Window")
            plt.ylabel("DeltaPPL (Window - Fix)")
            plt.title(f"{r1} vs {r2} | {fb} band")
            plt.legend(title="Class", loc="upper right", fontsize="small")
            plt.ylim(bottom=0)

            plt.tight_layout()
            fname = f"{r1.replace('/','_')}_{r2.replace('/','_')}_{fb}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
            plt.close()

    print(f"\nFinished plotting {total_plots} figures. Check '{OUTPUT_DIR}' for output.")
