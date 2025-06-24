import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
#  USER-CONFIGURABLE PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
DATA_DIR            = r"E:\data\project_repos\phzhr_turtles_av_ml\data\ppl"
REGION_PAIRS_CSV    = os.path.join(DATA_DIR, "region_pairs.csv")
FREQ_CSV            = os.path.join(DATA_DIR, "f1.csv")
FILE_SUFFIX         = "_ppl_iti.mat"
OUTPUT_DIR          = r"E:\data\project_repos\phzhr_turtles_av_ml\model_results\18_ppl_ribbon_pplxfreqbin"

# Map class indices → readable labels
CLASS_LABEL_MAP = {
    1: "Color",
    2: "Orientation",
    3: "Tone",
    4: "Duration"
}

# ───────────────────────────────────────────────────────────────────────────────
#  HELPER: load only the fields we need (combo labels, trialinfo, chancombo,
#           PPL_fix_all, PPL_dp_first1k)
# ───────────────────────────────────────────────────────────────────────────────
def load_subject_data(mat_path):
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    combo_labels = mat["combolabels"]        # shape: (nCombos, 3), dtype=object
    trialinfo    = mat["trialinfo"]          # shape: (nTrials, 8), dtype=float
    chancombo    = mat["chancombo"]          # shape: (nCombos, 2), dtype=float
    PPL_fix      = mat["PPL_fix_all"]        # shape: (nTrials, nCombos, nFreqs)
    PPL_dp1      = mat["PPL_dp_first1k"]     # shape: (nTrials, nCombos, nFreqs)
    return combo_labels, trialinfo, chancombo, PPL_fix, PPL_dp1

# ───────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Read region pairs: assume two‐column CSV with headers "Region1,Region2"
    df_pairs = pd.read_csv(REGION_PAIRS_CSV)
    region_pairs = [tuple(row) for row in df_pairs.values]

    # 2) Read frequency vector (56 bins) from f1.csv
    freqs = pd.read_csv(FREQ_CSV, header=None).values.ravel()

    # 3) Prepare containers for accumulating data
    classes       = [1, 2, 3, 4]
    data          = { (r1, r2): {cls: [] for cls in classes} for (r1, r2) in region_pairs }
    subj_set      = { (r1, r2): set()                  for (r1, r2) in region_pairs }
    combo_count   = { (r1, r2): 0                      for (r1, r2) in region_pairs }

    # 4) Find all subject‐session .mat files in DATA_DIR
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

    for fname in mat_files:
        subj = fname.split("_")[0]
        mat_path = os.path.join(DATA_DIR, fname)

        combo_labels, trialinfo, chancombo, PPL_fix, PPL_dp1 = load_subject_data(mat_path)

        # 4a) Filter out trials with any NaN in trialinfo
        valid_trials_mask = ~np.isnan(trialinfo).any(axis=1)
        if not valid_trials_mask.any():
            continue

        trialinfo_valid = trialinfo[valid_trials_mask, :]
        trial_classes   = trialinfo_valid[:, 6].astype(int)   # 7th column = class ∈ {1..4}

        PPL_fix_valid = PPL_fix[valid_trials_mask, :, :]       # shape: (nValid, nCombos, nFreqs)
        PPL_dp1_valid = PPL_dp1[valid_trials_mask, :, :]       # same shape

        # 4b) Compute ΔPPL for DP1 relative to Fixation: (nValid, nCombos, nFreqs)
        dPPL_dp1 = PPL_dp1_valid - PPL_fix_valid
        nValid, nCombos, nFreqs = dPPL_dp1.shape

        # 4c) For each region‐pair, find matching combo indices & accumulate
        for (r1, r2) in region_pairs:
            concat1 = r1 + r2
            concat2 = r2 + r1

            # combo_labels[i][2] is the concatenated ROI string for combo i
            combo_idxs = [
                i for i in range(nCombos)
                if (str(combo_labels[i][2]) == concat1) or (str(combo_labels[i][2]) == concat2)
            ]
            if not combo_idxs:
                continue

            combo_idxs = np.array(combo_idxs, dtype=int)
            subj_set[(r1, r2)].add(subj)
            combo_count[(r1, r2)] += combo_idxs.size

            # 4d) For each class: average across trials, then store per‐combo ΔPPL
            for cls in classes:
                trial_idxs = np.where(trial_classes == cls)[0]
                if trial_idxs.size == 0:
                    continue

                # submat: shape (nTrials_cls, nCombos_pair, nFreqs)
                submat = dPPL_dp1[trial_idxs][:, combo_idxs, :]

                # mean_per_combo: shape (nCombos_pair, nFreqs)
                with np.errstate(invalid="ignore"):
                    mean_per_combo = np.nanmean(submat, axis=0)

                # append each combo's frequency‐vector (if not all‐NaN) to the list
                for combo_vec in mean_per_combo:
                    if not np.all(np.isnan(combo_vec)):
                        data[(r1, r2)][cls].append(combo_vec)

    # 5) Generate ribbon plot per region‐pair
    for (r1, r2) in region_pairs:
        n_subj  = len(subj_set[(r1, r2)])
        n_combo = combo_count[(r1, r2)]

        # skip plotting if no class has any data
        if all(len(data[(r1, r2)][cls]) == 0 for cls in classes):
            continue

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        for cls in classes:
            class_list = data[(r1, r2)][cls]
            if len(class_list) == 0:
                continue

            arr = np.vstack(class_list)   # shape: (nCombos_total_for_cls, nFreqs)
            mean_vec = np.nanmean(arr, axis=0)
            sem_vec  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

            ax.plot(freqs, mean_vec, label=CLASS_LABEL_MAP[cls])
            ax.fill_between(freqs,
                            mean_vec - sem_vec,
                            mean_vec + sem_vec,
                            alpha=0.3)

        # display counts in upper‐left
        ax.text(0.02, 0.98,
                f"n_subj={n_subj}, n_combo={n_combo}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ΔPPL (DP1 − Fix)")
        ax.set_title(f"{r1} vs {r2}")
        ax.legend(title="Class", fontsize="small", loc="upper right")
        ax.grid(False)

        plt.tight_layout()
        save_name = f"{r1.replace('/', '_')}_{r2.replace('/', '_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150)
        plt.close()
