# -*- coding: utf-8 -*-
"""
Composite ribbon plots per ROI group
Created on Tue May 13 11:57:43 2025
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
    '08_noholdout_evalallperiods',
    'DPTRAIN_evalfixencdp_minirocket_channel_val_accuracies.csv'
)

# Load your results
df = pd.read_csv(csv_path)

# Constants for plotting
timepoints = ['FixationAccuracy', 'EncodingAccuracy', 'DelayAccuracy']
labels     = ['Fixation', 'Encoding', 'Delay']
x_pos      = np.arange(len(timepoints))

# Define ROI groups
roi_groups = {
    "R Temporal ROIs Trial Classification Accuracy": [
        'R Superior Temporal Gyrus', 'R ParaHippocampal Gyrus',
        'R Middle Temporal Gyrus', 'R Medial Temporal Pole',
        'R Inferior Temporal Gyrus', 'R Fusiform Gyrus', 'R Hippocampus'
    ],
    "L Temporal ROIs Trial Classification Accuracy": [
        'L Fusiform Gyrus', 'Left Amygdala', 'L Superior Temporal Gyrus',
        'L ParaHippocampal Gyrus', 'L Middle Temporal Gyrus',
        'L Medial Temporal Pole', 'L Inferior Temporal Gyrus',
        'L Hippocampus', 'L Heschls Gyrus', 'L Amygdala'
    ],
    "R Frontal ROIs Trial Classification Accuracy": [
        'R Middle Frontal Gyrus', 'R IFG (p Triangularis)'
    ],
    "L Frontal ROIs Trial Classification Accuracy": [
        'L Superior Orbital Gyrus', 'L Precentral Gyrus', 'L Postcentral Gyrus',
        'L Middle Frontal Gyrus', 'L IFG (p Triangularis)',
        'L IFG (p Orbitalis)', 'L IFG (p Opercularis)'
    ]
}

# Generate one plot per group
for title, rois in roi_groups.items():
    fig, ax = plt.subplots(figsize=(8,5))
    
    # pick a distinct color per ROI
    colors = plt.cm.tab10(np.linspace(0,1,len(rois)))
    
    for color, roi in zip(colors, rois):
        sub = df[df['ROI'] == roi]
        if sub.empty:
            continue
        
        # mean & SEM across channels
        means = sub[timepoints].mean()
        sems  = sub[timepoints].std(ddof=1) / np.sqrt(len(sub))
        n_chan = len(sub)  # number of channels
        
        # plot line + ribbon
        ax.plot(x_pos, means, marker='o', color=color,
                label=f"{roi} (n={n_chan})")
        ax.fill_between(x_pos,
                        means - sems,
                        means + sems,
                        color=color, alpha=0.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title(title)
    
    # legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    plt.show()
