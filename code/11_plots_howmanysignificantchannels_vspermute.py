#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025

Plot number of significant channels (Delay_p_perm < .05) per ROI & class,
plus an “Any” bar and n-subject counts, with custom log-scale yticks.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ── PARAMETERS ────────────────────────────────────────────────────────────────
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir,
    '11_permutationshuffledtrain',
    '11_onevsall_with_permutationontrain.csv'
)
alpha = 0.05  # p-value threshold for Delay_p_perm
# ── END PARAMETERS ────────────────────────────────────────────────────────────

# 1) Load CSV
df = pd.read_csv(csv_path)

# 2) Subjects per ROI (for the "n=" annotation)
df_subj = df[['Subject', 'ROI']].drop_duplicates()
roi_subj_counts = df_subj.groupby('ROI')['Subject'].nunique().to_dict()

# 3) Total channels per ROI (background bars)
total_per_roi = (
    df[['ROI', 'ChannelIndex']]
    .drop_duplicates()
    .groupby('ROI')['ChannelIndex']
    .nunique()
    .to_dict()
)

# 4) Filter to significant delay-period channels
df_sig = df[df['Delay_p_perm'] < alpha]

# 5) Count significant channels per ROI × ClassLabel
sig_counts = (
    df_sig
    .groupby(['ROI', 'ClassLabel'])['ChannelIndex']
    .nunique()
    .reset_index(name='Count')
)
sig_pivot = sig_counts.pivot(index='ROI', columns='ClassLabel', values='Count').fillna(0)

# 6) “Any” significant per ROI (channel counted once even if in multiple classes)
any_sig = (
    df_sig[['ROI','ChannelIndex']]
    .drop_duplicates()
    .groupby('ROI')['ChannelIndex']
    .nunique()
    .rename('Any')
)
sig_pivot['Any'] = any_sig

# 7) Prepare plotting order
rois = sorted(
    total_per_roi.keys(),
    key=lambda r: -total_per_roi[r]
)
classes = list(sig_pivot.columns)  # e.g. ['Colors','Orientation','Tone','Noise Duration','Any']

# 8) Plot
x = np.arange(len(rois))
width = 0.8
inner = width / len(classes) * 0.9
palette = sns.color_palette('tab10', len(classes))

plt.figure(figsize=(18, 6), dpi=150)

# 8a) Background = total channels
plt.bar(x, [total_per_roi[r] for r in rois],
        width=width, color='lightgrey', label='Total channels')

# 8b) Overlay = significant counts per class + Any
for i, cls in enumerate(classes):
    heights = [sig_pivot.loc[r, cls] if r in sig_pivot.index else 0 for r in rois]
    offset = (i - (len(classes)-1)/2) * inner
    plt.bar(x + offset, heights,
            width=inner, label=cls, color=palette[i])

# 8c) Annotate n-subjects above each ROI
for xi, roi in enumerate(rois):
    n = roi_subj_counts.get(roi, 0)
    plt.text(xi, total_per_roi[roi] * 1.05, f"n={n}", ha='center')

# 9) Formatting
plt.yscale('log')
# custom ticks all the way up
yticks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
plt.yticks(yticks, [str(y) for y in yticks])
plt.ylabel('# of Significant Channels (Delay p<.05, log scale)')
plt.xlabel('ROI')
plt.xticks(
    x,
    [r if len(r) <= 15 else r[:15] + '...' for r in rois],
    rotation=45, ha='right'
)
plt.title('Delay-Period Significant Channels per ROI & Class')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()