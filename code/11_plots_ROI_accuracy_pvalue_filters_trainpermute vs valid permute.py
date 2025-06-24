import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Update these roots to your local paths ===
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
old_csv = os.path.join(
    model_results_dir,
    '10_permutationsigtest',
    '10_onevsall_with_paired_signif.csv'
)
new_csv = os.path.join(
    model_results_dir,
    '11_permutationshuffledtrain',
    '11_onevsall_with_permutationontrain.csv'
)
ieeg_root = r'E:\data\k12wm'

# Load both result tables
df_old = pd.read_csv(old_csv)
df_new = pd.read_csv(new_csv)

# Pre-load gamma-mod channels map (unchanged from before)
gamma_map = {}
for subj in df_new['Subject'].unique():
    sess_dir = f"{subj}_turtles_s01"
    gm_path  = os.path.join(ieeg_root, subj, sess_dir, f"{sess_dir}gammamodchans.mat")
    subj_map = {}
    try:
        gm   = scipy.io.loadmat(gm_path)
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
timepoints      = ['FixationAccuracy','EncodingAccuracy','DelayAccuracy']
labels          = ['Fixation','Encoding','Delay']
x_pos           = np.arange(len(timepoints))
class_label_map = {1:"Color",2:"Orientation",3:"Tone",4:"Duration"}
class_labels    = sorted(df_new['ClassLabel'].unique())

# Unique ROIs
all_rois = sorted(df_new['ROI'].dropna().unique())

# Output directory
out_compare = os.path.join(
    model_results_dir,
    '11_permutationshuffledtrain',
    'roi_comparison_plots_2x2'
)
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
    handles, lbls = ax.get_legend_handles_labels()
    if lbls:
        ax.legend(handles, lbls, loc='upper right', fontsize='small')

for roi in all_rois:
    # all‐channels from the new table (same channels)
    df_all = df_new[df_new['ROI']==roi]
    if df_all.empty:
        continue
    n_all = df_all['Subject'].nunique()

    # old‐style perm‐sig
    df_old_roi  = df_old[df_old['ROI']==roi]
    df_old_sig  = df_old_roi[df_old_roi['Delay_signif_perm']]
    n_old       = df_old_sig['Subject'].nunique()

    # new‐style perm‐sig
    df_new_sig  = df_all[df_all['Delay_signif_perm']]
    n_new       = df_new_sig['Subject'].nunique()

    # 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    fig.suptitle(roi, fontsize=20)
    plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3)

    # Top‐left: All channels
    plot_ribbon(axes[0,0], df_all,   "All Channels",           n_all)
    # Top‐right: Old‐style permutation
    plot_ribbon(axes[0,1], df_old_sig, "Old Perm-sig Channels", n_old)
    # Bottom‐left: New‐style permutation-on-train
    plot_ribbon(axes[1,0], df_new_sig, "New Perm-train Channels", n_new)
    # Bottom-right: unused
    axes[1,1].axis('off')

    # show & save
    plt.show()
    fname = roi.replace('/','_').replace('\\','_')
    save_path = os.path.join(out_compare, f"{fname}_2x2.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
