#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:43:47 2025

@author: phzhr
"""
import os
import re
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the combined .mat files
data_dir = '/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/data'
file_suffix = '_uV_allregions.mat'
mat_files = [f for f in os.listdir(data_dir) if f.endswith(file_suffix)]

#Build per-subject per-ROI per-task records, plus "Any" aggregated
records = []
for fname in mat_files:
    subj = fname.split('_')[0]
    mat = scipy.io.loadmat(os.path.join(data_dir, fname))
    anats = mat['anatomicallabels']
    rois = [re.sub(r"[\[\]']", "", str(r[1])).strip() for r in anats]
    sig1 = mat['sigchans1'].flatten()
    sig2 = mat['sigchans2'].flatten()
    # compute union for "Any"
    union_all = np.unique(np.concatenate([np.intersect1d(np.atleast_1d(sig1[i]).ravel(),
                                                        np.atleast_1d(sig2[i]).ravel())
                                           for i in range(4)]))
    unique_rois = sorted(set(rois))
    for roi in unique_rois:
        ch_idx = [i for i,r in enumerate(rois) if r==roi]
        total = len(ch_idx)
        # for each task
        for t in range(4):
            ov = np.intersect1d(np.atleast_1d(sig1[t]).ravel(), np.atleast_1d(sig2[t]).ravel())
            gamma = sum(((i+1) in ov) for i in ch_idx)
            records.append({'Subject':subj,'ROI':roi,'Task':f'T{t+1}',
                            'Total':total,'Gamma':gamma})
        # Any
        gamma_any = sum(((i+1) in union_all) for i in ch_idx)
        records.append({'Subject':subj,'ROI':roi,'Task':'Any',
                        'Total':total,'Gamma':gamma_any})

df = pd.DataFrame(records)

## SET UP PLOT
# ROI subject counts
df_subj = df[['Subject','ROI']].drop_duplicates()
roi_subj_counts = df_subj.groupby('ROI').size()

# # Quick QA: print any blank ROI labels with subject & channel index, they should be due to empty channels
# print("Checking for unlabeled (blank) ROIs&")
# for fname in mat_files:
#     subj = fname.split('_')[0]
#     mat = scipy.io.loadmat(os.path.join(data_dir, fname))
#     anats = mat['anatomicallabels']
#     # clean ROI names exactly as you do downstream
#     rois = [re.sub(r"[\[\]']", "", str(r[1])).strip() for r in anats]
#     for ch_idx, roi in enumerate(rois, start=1):  # 1-based channel numbering
#         if roi == "":
#             print(f"  Blank ROI for subject {subj}, channel {ch_idx}")

df_agg = df.groupby(['ROI', 'Task'])[['Total', 'Gamma']].sum().reset_index()

# Compute total channels and "Any" gamma counts per ROI (aggregated)
roi_totals = df_agg.groupby('ROI')['Total'].max().to_dict()
any_counts = df_agg[df_agg.Task == 'Any'].set_index('ROI')['Gamma'].to_dict()

# Count number of subjects per ROI (from original per-subject df)
df_subj = df[['Subject', 'ROI']].drop_duplicates()
roi_subj_counts = df_subj.groupby('ROI').size()

# Filter ROIs
rois_keep = [
    roi for roi in roi_totals
    if roi_subj_counts.get(roi, 0) >= 4
    and roi not in ('Unknown', '')
    and any_counts.get(roi, 0) > 1
]

#Filter aggregated data
df_agg = df_agg[df_agg.ROI.isin(rois_keep)]
roi_totals = df_agg.groupby('ROI')['Total'].max().to_dict()
any_counts = df_agg[df_agg.Task == 'Any'].set_index('ROI')['Gamma'].to_dict()
roi_subj_counts = df[df.ROI.isin(rois_keep)][['Subject', 'ROI']].drop_duplicates().groupby('ROI').size()
# Sort ROIs by (subjects desc, any_gamma/total desc)
sorted_rois = sorted(
    rois_keep,
    key=lambda roi: -roi_totals[roi]
)

# Setup plotting
tasks = ['T1','T2','T3','T4','Any']
label_map = {
    'T1': 'Colors',
    'T2': 'Orientation',
    'T3': 'Tone',
    'T4': 'Noise Duration',
    'Any': 'Any'
}

x = np.arange(len(sorted_rois))
width = 0.8; inner = width/len(tasks)*0.9
palette = sns.color_palette('tab10', len(tasks))

plt.figure(figsize=(18,6), dpi=150)
# Background bars
plt.bar(x, [roi_totals[r] for r in sorted_rois], width=width, color='lightgrey')

# Task bars
for i, t in enumerate(tasks):
    heights = [df_agg[(df_agg.ROI==r)&(df_agg.Task==t)]['Gamma'].sum() for r in sorted_rois]
    offset = (i - (len(tasks)-1)/2)*(width/len(tasks))
    plt.bar(x+offset, heights, width=inner, color=palette[i], label=label_map[t])

# Annotate n-subjects
for xi, roi in enumerate(sorted_rois):
    plt.text(xi, roi_totals[roi]*1.05, f"n={roi_subj_counts[roi]}", ha='center')

# Log scale
plt.yscale('log')
plt.yticks([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], labels=[str(i) for i in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]])
plt.ylabel('# of Channels (log scale)')
plt.xticks(x, [r if len(r)<=15 else r[:15]+'...' for r in sorted_rois],
           rotation=45, ha='right')
plt.xlabel('ROI')
plt.title('Gamma-modulated Channels per ROI across 13 sessions')
plt.legend(title='Task', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()