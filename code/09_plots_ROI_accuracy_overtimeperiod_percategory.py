# -*- coding: utf-8 -*-
"""
Composite ribbon plots per ROI — now grouped by ROI with ribbon lines for each class (L=1–4)
Created on Tue May 14 2025
@author: phzhr
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Update this to your local path ===
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir,
    '09_onevsall',
    'ENCTRAIN_noholdout_onevsall.csv'
)

# Load results
df = pd.read_csv(csv_path)

# Constants for plotting
timepoints = ['FixationAccuracy', 'EncodingAccuracy', 'DelayAccuracy']
labels     = ['Fixation', 'Encoding', 'Delay']
x_pos      = np.arange(len(timepoints))

# Class label names
class_label_map = {
    1: "Color",
    2: "Orientation",
    3: "Tone",
    4: "Duration"
}

# All unique ROIs in your data
all_rois = sorted(df['ROI'].dropna().unique())
class_labels = sorted(df['ClassLabel'].unique())

# Directory to save plots
save_dir = os.path.join(model_results_dir, '09_onevsall', 'all_channels_ribbon_by_roi')
os.makedirs(save_dir, exist_ok=True)  # make dir if not exist

# Generate one plot per ROI
for roi in all_rois:
    roi_df = df[df['ROI'] == roi]
    if roi_df.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color per class label
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels)))

    for color, L in zip(colors, class_labels):
        sub = roi_df[roi_df['ClassLabel'] == L]
        if sub.empty:
            continue

        means = sub[timepoints].mean()
        sems = sub[timepoints].std(ddof=1) / np.sqrt(len(sub))
        n_chan = len(sub)

        class_name = class_label_map.get(L, f"Class {L}")
        label = f"{class_name} (n={n_chan})"
        ax.plot(x_pos, means, marker='o', color=color, label=label)
        ax.fill_between(x_pos, means - sems, means + sems, color=color, alpha=0.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title(f'{roi} — Trial Classification Accuracy by Class')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Save figure
    clean_roi = roi.replace('/', '_').replace('\\', '_')  # sanitize filename
    save_path = os.path.join(save_dir, f'{clean_roi} All Channels.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)  # close to avoid GUI pop-up