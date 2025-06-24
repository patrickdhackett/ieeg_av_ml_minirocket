import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Update these roots to your local paths ===
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir,
    '11_permutationshuffledtrain/11_onevsall_with_permutationontrain.csv'
)
ieeg_root = r'E:\data\k12wm'

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

# Plot settings
timepoints = ['FixationAccuracy','EncodingAccuracy','DelayAccuracy']
labels     = ['Fixation','Encoding','Delay']
x_pos      = np.arange(len(timepoints))
class_label_map = {1:"Color",2:"Orientation",3:"Tone",4:"Duration"}
class_labels = sorted(df['ClassLabel'].unique())

# Unique ROIs
all_rois = sorted(df['ROI'].dropna().unique())

# Output directory
out_compare = os.path.join(model_results_dir, '11_permutationshuffledtrain', 'roi_comparison_plots_wnull')
os.makedirs(out_compare, exist_ok=True)

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
    # only draw legend if there are labeled artists
    handles, lbls = ax.get_legend_handles_labels()
    if lbls:
        ax.legend(handles, lbls, loc='upper right', fontsize='small')

# Null plot settings
timepoints_null = ['Fix_null_mean','Enc_null_mean','Delay_null_mean']
def plot_ribbon_null(ax, df_sub, title, n_subj):
    colors = plt.cm.tab10(np.linspace(0,1,len(class_labels)))
    for color, L in zip(colors, class_labels):
        sub = df_sub[df_sub['ClassLabel']==L]
        if sub.empty:
            continue
        means = sub[timepoints_null].mean()
        sems  = sub[timepoints_null].std(ddof=1)/np.sqrt(len(sub))
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
    # only draw legend if there are labeled artists
    handles, lbls = ax.get_legend_handles_labels()
    if lbls:
        ax.legend(handles, lbls, loc='upper right', fontsize='small')
# ── PLOTTING: 3 on top, 2 on bottom ──────────────────────────────────────────

for roi in all_rois:
    df_roi = df[df['ROI']==roi]
    if df_roi.empty:
        continue

    # 1) All channels
    df_all = df_roi
    n_all  = df_all['Subject'].nunique()

    # 2) Gamma-modulated channels
    mask_g = df_roi.apply(
        lambda r: r['ChannelIndex'] in gamma_map.get(r['Subject'], {}).get(int(r['ClassLabel']), []),
        axis=1
    )
    df_gamma = df_roi[mask_g]
    n_gamma  = df_gamma['Subject'].nunique()

    # 3) Delay-vs-Fixation significant (paired perm test)
    df_dpfx = df_roi[df_roi['DpFx_signif_perm']]
    n_dpfx  = df_dpfx['Subject'].nunique()

    # 4) Delay period binomial-significant
    df_dpbin = df_roi[df_roi['Delay_signif_binom']]
    n_dpbin  = df_dpbin['Subject'].nunique()

    # 5) Delay period permutation-significant
    df_dpperm = df_roi[df_roi['Delay_signif_perm']]
    n_dpperm  = df_dpperm['Subject'].nunique()
    
    # 6) Delay period permutation-significant
    df_perm = df_roi
    n_perm = df_all['Subject'].nunique()
    
    # create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    fig.suptitle(roi, fontsize=18)
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

    # top row
    plot_ribbon(axes[0, 0], df_all,    "All Channels",                  n_all)
    plot_ribbon(axes[0, 1], df_gamma,  "Gamma-modulated Channels",      n_gamma)
    plot_ribbon(axes[0, 2], df_dpfx,   "Delay vs Fixation Significant", n_dpfx)

    # bottom row
    plot_ribbon(axes[1, 0], df_dpbin,  "Delay Binomial Significant",    n_dpbin)
    plot_ribbon(axes[1, 1], df_dpperm, "Delay Permutation Significant", n_dpperm)
    plot_ribbon_null(axes[1, 2], df_perm, "Null Shuffled Permutation", n_perm)

    # save and display
    fname = roi.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(out_compare, f"{fname}_6grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
