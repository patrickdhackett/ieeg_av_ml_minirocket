import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Update these roots ───────────────────────────────────────────────────────
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir,
    '12_no_permutation',
    '12_onevsall_noPerm_with_multipleDelays.csv'
)
ieeg_root = r'E:\data\k12wm'
out_compare = os.path.join(model_results_dir, '12_no_permutation', 'roi_comparison_plots')
os.makedirs(out_compare, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────

# Load results
df = pd.read_csv(csv_path)

# Pre-load gamma-modulated channels per subject, per class
gamma_map = {}
for subj in df['Subject'].unique():
    sess_dir = f"{subj}_turtles_s01"
    gm_path  = os.path.join(ieeg_root, subj, sess_dir, f"{sess_dir}gammamodchans.mat")
    subj_map = {}
    try:
        gm = scipy.io.loadmat(gm_path)
        sig1 = gm['sigchans1']
        sig2 = gm['sigchans2']
        for L in range(1,5):
            ch1 = np.atleast_1d(sig1[L-1][0]).astype(int).flatten()
            ch2 = np.atleast_1d(sig2[L-1][0]).astype(int).flatten()
            subj_map[L] = sorted(set(ch1.tolist() + ch2.tolist()))
    except Exception:
        subj_map = {1:[],2:[],3:[],4:[]}
    gamma_map[subj] = subj_map

# Plot settings: five epochs
timepoints = [
    'FixationAccuracy',
    'EncodingAccuracy',
    'Delay1Accuracy',
    'Delay2Accuracy',
    'Delay3Accuracy',
]
labels = ['Fix','Enc','D1','D2','D3']
x_pos = np.arange(len(timepoints))

class_label_map = {1:"Color",2:"Orientation",3:"Tone",4:"Duration"}
class_labels = sorted(df['ClassLabel'].astype(int).unique())

def plot_ribbon(ax, df_sub, title, n_subj):
    colors = plt.cm.tab10(np.linspace(0,1,len(class_labels)))
    for color, L in zip(colors, class_labels):
        sub = df_sub[df_sub['ClassLabel']==L]
        if sub.empty: 
            continue
        means = sub[timepoints].mean()
        sems  = sub[timepoints].std(ddof=1)/np.sqrt(len(sub))
        lbl   = f"{class_label_map[L]} (n={len(sub)})"
        ax.plot(x_pos, means, marker='o', color=color, label=lbl)
        ax.fill_between(x_pos, means-sems, means+sems, color=color, alpha=0.2)
    ax.axhline(0.5, linestyle='--', color='gray')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title(f"{title}\n(n subj={n_subj})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    h,l = ax.get_legend_handles_labels()
    if l:
        ax.legend(h, l, loc='upper right', fontsize='small')

# Unique ROIs
all_rois = sorted(df['ROI'].dropna().unique())

for roi in all_rois:
    df_roi = df[df['ROI']==roi]
    if df_roi.empty:
        continue

    # 1) All channels
    df_all   = df_roi
    n_all    = df_all['Subject'].nunique()

    # 2) Gamma‐modulated channels
    mask_g   = df_roi.apply(
        lambda r: r['ChannelIndex'] in gamma_map.get(r['Subject'], {}).get(int(r['ClassLabel']), []),
        axis=1
    )
    df_gamma = df_roi[mask_g]
    n_gamma  = df_gamma['Subject'].nunique()

    # 3) Any DP window vs Fix significant (paired McNemar)
    sig_diff_mask = (
        (df_roi['Delay1_vsFix_p'] < 0.05) |
        (df_roi['Delay2_vsFix_p'] < 0.05) |
        (df_roi['Delay3_vsFix_p'] < 0.05)
    )
    df_diff = df_roi[sig_diff_mask]
    n_diff  = df_diff['Subject'].nunique()

    # 4) Any DP window binomially significant
    sig_binom_mask = (
        df_roi['Delay1_signif_binom'] |
        df_roi['Delay2_signif_binom'] |
        df_roi['Delay3_signif_binom']
    )
    df_binom = df_roi[sig_binom_mask]
    n_binom  = df_binom['Subject'].nunique()

    # 2×2 figure
    fig, axes = plt.subplots(2,2, figsize=(14,10), sharey=True)
    fig.suptitle(roi, fontsize=18)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plot_ribbon(axes[0,0], df_all,   "1) All Channels",      n_all)
    plot_ribbon(axes[0,1], df_gamma, "2) Gamma‐mod Channels", n_gamma)
    plot_ribbon(axes[1,0], df_diff,  "3) D vs Fix Significant", n_diff)
    plot_ribbon(axes[1,1], df_binom, "4) Any Delay Sig Binom", n_binom)

    # Save
    fname = roi.replace('/','_').replace('\\','_')
    path  = os.path.join(out_compare, f"{fname}_4grid.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
