import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Update these roots to your local paths ===
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir, '09_onevsall', 'ENCTRAIN_noholdout_onevsall.csv'
)
ieeg_root = r'E:\data\k12wm'

# Load results
df = pd.read_csv(csv_path)

# Pre‐load gamma‐modulated channels per subject, per class
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
    except Exception as e:
        print(f"⚠️ Could not load gamma channels for {subj}: {e}")
    gamma_map[subj] = subj_map

# Plot settings
timepoints = ['FixationAccuracy', 'EncodingAccuracy', 'DelayAccuracy']
labels     = ['Fixation', 'Encoding', 'Delay']
x_pos      = np.arange(len(timepoints))
class_label_map = {1:"Color",2:"Orientation",3:"Tone",4:"Duration"}
class_labels = sorted(df['ClassLabel'].unique())

# Unique ROIs
all_rois = sorted(df['ROI'].dropna().unique())

# Output directory
out_compare = os.path.join(model_results_dir, '09_onevsall', 'comparison_ribbon_by_roi')
os.makedirs(out_compare, exist_ok=True)

# Function to plot on a given axis
def plot_ribbon(ax, df_sub, suffix, n_subj):
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
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"{suffix} (n subjects={n_subj})")
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

# Generate comparison plots
for roi in all_rois:
    df_roi = df[df['ROI']==roi]
    if df_roi.empty:
        continue
    # count unique subjects in this ROI
    n_subj = df_roi['Subject'].nunique()
    
    # gamma-only filter
    mask = df_roi.apply(
        lambda r: r['ChannelIndex'] in gamma_map.get(r['Subject'], {}).get(int(r['ClassLabel']), []),
        axis=1
    )
    df_gamma = df_roi[mask]
    
    # create figure with two subplots
    fig, axes = plt.subplots(1,2,figsize=(12,5), gridspec_kw={'width_ratios':[1,1]})
    
    # 1) add the overall ROI name
    fig.suptitle(roi, fontsize=18)
    fig.subplots_adjust(top=0.85)

    # all channels
    plot_ribbon(axes[0], df_roi, "All Channels", n_subj)
    # gamma-only
    if df_gamma.empty:
        axes[1].text(0.5,0.5,"No gamma‐modulated data", ha='center', va='center')
        axes[1].set_axis_off()
    else:
        plot_ribbon(axes[1], df_gamma, "Gamma‐modulated Channels", n_subj)
    
    plt.tight_layout()
    #fname = roi.replace('/','_').replace('\\','_')
   # save_path = os.path.join(out_compare, f"{fname}_comparison.png")
    plt.show()
    # plt.savefig(save_path, dpi=150)
    plt.close(fig)
