#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-panel ROI-level comparison of real vs. null accuracies
(“_iti” one-vs-all permutation results with train/test split),
with Fix/Enc/Delay1/Delay2 on the x-axis and proper null-SEM pooling.
"""

import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── paths & parameters ─────────────────────────────────────────────────────
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
csv_path = os.path.join(
    model_results_dir,
    '15_permutationshuffledtrain_iti_testsplit',
    '15_onevsall_with_permutationontrain_iti_testsplit_20percent.csv'
)
ieeg_root = r'E:\data\k12wm'
ALPHA     = 0.05

# ─── load results & build gamma-map ─────────────────────────────────────────
df = pd.read_csv(csv_path)

gamma_map = {}
for subj in df['Subject'].unique():
    sess = f"{subj}_turtles_s01"
    gm_path = os.path.join(ieeg_root, subj, sess, f"{sess}gammamodchans.mat")
    try:
        gm   = scipy.io.loadmat(gm_path)
        sig1, sig2 = gm['sigchans1'], gm['sigchans2']
        m = {}
        for L in (1,2,3,4):
            c1 = np.atleast_1d(sig1[L-1][0]).astype(int).ravel()
            c2 = np.atleast_1d(sig2[L-1][0]).astype(int).ravel()
            m[L] = sorted(set(c1.tolist()+c2.tolist()))
    except:
        m = {1:[],2:[],3:[],4:[]}
    gamma_map[subj] = m

# ─── epoch definitions ───────────────────────────────────────────────────────
timepoints   = ['FixationAccuracy','EncodingAccuracy',
                'Delay1Accuracy','Delay2Accuracy']
labels       = ['Fixation','Encoding','Delay1','Delay2']
x_pos        = np.arange(len(timepoints))

null_means   = ['Fix_null_mean','Enc_null_mean',
                'D1_null_mean','D2_null_mean']
null_stds    = ['Fix_null_std','Enc_null_std',
                'D1_null_std','D2_null_std']

class_label_map = {1:"Color",2:"Orientation",3:"Tone",4:"Duration"}
class_labels    = sorted(df['ClassLabel'].unique())

out_compare = os.path.join(
    model_results_dir,
    '15_permutationshuffledtrain_iti_testsplit',
    'roi_comparison_plots_withdotted'
)
os.makedirs(out_compare, exist_ok=True)

# ─── plotting helpers ───────────────────────────────────────────────────────
def plot_ribbon(ax, dsub, title, nsubj):
    colors = plt.cm.tab10(np.linspace(0,1,len(class_labels)))
    for c,L in zip(colors,class_labels):
        sub = dsub[dsub['ClassLabel']==L]
        if sub.empty: continue
        means = sub[timepoints].mean().values
        sems  = sub[timepoints].std(ddof=1).values/np.sqrt(len(sub))
        lbl   = f"{class_label_map[L]} (n={len(sub)})"
        ax.plot(x_pos, means, '-o', color=c, label=lbl)
        ax.fill_between(x_pos, means-sems, means+sems, color=c, alpha=0.2)
    ax.axhline(0.5, ls='--', color='gray')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title(f"{title}\n(n subj={nsubj})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    h,l = ax.get_legend_handles_labels()
    if l: ax.legend(h,l,loc='upper right',fontsize='small')

def plot_null(ax, dsub, title, nsubj):
    colors = plt.cm.tab10(np.linspace(0,1,len(class_labels)))
    for c,L in zip(colors,class_labels):
        sub = dsub[dsub['ClassLabel']==L]
        if sub.empty: continue
        nm  = sub[null_means].mean().values   # 4 null-means
        ns  = sub[null_stds].mean().values    # 4 null-stds
        sem = ns/np.sqrt(len(sub))
        lbl = f"{class_label_map[L]} (n={len(sub)})"
        ax.plot(x_pos, nm, '-o', color=c, label=lbl)
        ax.fill_between(x_pos, nm-sem, nm+sem, color=c, alpha=0.2)
    ax.axhline(0.5, ls='--', color='gray')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title(f"{title}\n(n subj={nsubj})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    h,l = ax.get_legend_handles_labels()
    if l: ax.legend(h,l,loc='upper right',fontsize='small')

def plot_ribbon_perm(ax, dsub, title, nsubj, classes_missing):
    colors = plt.cm.tab10(np.linspace(0,1,len(class_labels)))
    for c,L in zip(colors,class_labels):
        sub = dsub[dsub['ClassLabel']==L]
        if sub.empty:
            continue
        means = sub[timepoints].mean().values
        sems  = sub[timepoints].std(ddof=1).values/np.sqrt(len(sub))
        lbl   = f"{class_label_map[L]} (n={len(sub)})"
        style = '--' if L in classes_missing else '-'
        ax.plot(x_pos, means, marker='o', linestyle=style, color=c, label=lbl)
        ax.fill_between(x_pos, means-sems, means+sems, color=c, alpha=0.2)
    ax.axhline(0.5, ls='--', color='gray')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title(f"{title}\n(n subj={nsubj})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    h,l = ax.get_legend_handles_labels()
    if l: ax.legend(h,l,loc='upper right',fontsize='small')

# ─── main ROI loop ─────────────────────────────────────────────────────────
for roi in sorted(df['ROI'].dropna().unique()):
    d_roi = df[df['ROI']==roi]
    if d_roi.empty:
        continue

    # 1) all channels
    df_all   = d_roi
    n_all    = df_all['Subject'].nunique()

    # 2) gamma-modulated
    df_gam   = d_roi[d_roi.apply(
        lambda r: r['ChannelIndex'] in gamma_map[r['Subject']][int(r['ClassLabel'])],
        axis=1
    )]
    n_gam    = df_gam['Subject'].nunique()

    # 3) Delay1 vs Fix perm-sig
    df_dp1fx = d_roi[d_roi['D1_signif_vsFix']]
    n_dp1fx  = df_dp1fx['Subject'].nunique()

    # 4) Delay1 binomial-sig
    df_dp1bin= d_roi[d_roi['D1_signif_binom']]
    n_dp1bin = df_dp1bin['Subject'].nunique()

    # 5) Delay1 perm p<.05 (raw)
    df_dp1perm = d_roi[d_roi['D1_p_perm'] < ALPHA]
    n_dp1perm  = df_dp1perm['Subject'].nunique()

    # detect any missing classes & fill in
    present = set(df_dp1perm['ClassLabel'])
    missing = [L for L in class_labels if L not in present]

    if missing:
        ch_sets = [
            set(df_dp1perm[df_dp1perm['ClassLabel']==L]['ChannelIndex'])
            for L in present
        ]
        common_ch = set.intersection(*ch_sets) if ch_sets else set()
        fill_ins = []
        for L in missing:
            fill_ins.append(df_all[
                (df_all['ClassLabel']==L) &
                (df_all['ChannelIndex'].isin(common_ch))
            ])
        df_fake = pd.concat(fill_ins, ignore_index=True) if fill_ins else pd.DataFrame(columns=df_dp1perm.columns)
        df_dp1perm_complete = pd.concat([df_dp1perm, df_fake], ignore_index=True)
        n_dp1perm_complete = df_dp1perm_complete['Subject'].nunique()
    else:
        df_dp1perm_complete = df_dp1perm
        n_dp1perm_complete = n_dp1perm

    # 6) null (all channels)
    df_null  = d_roi
    n_null   = df_null['Subject'].nunique()

    # ─── draw 2×3 grid ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2,3, figsize=(18,10), sharey=True)
    fig.suptitle(roi, fontsize=18)
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

    # top row
    plot_ribbon(     axes[0,0], df_all,           "All channels",            n_all)
    plot_ribbon(     axes[0,1], df_gam,           "Gamma-modulated channels",n_gam)
    plot_ribbon(     axes[0,2], df_dp1fx,         "Delay1 vs Fix (perm-sig)",n_dp1fx)

    # bottom row
    plot_ribbon(     axes[1,0], df_dp1bin,        "Delay1 (binom-sig)",      n_dp1bin)
    plot_ribbon_perm(axes[1,1], df_dp1perm_complete,
                                             "Delay1 (perm p<0.05)",
                                             n_dp1perm_complete,
                                             classes_missing=missing)
    plot_null(       axes[1,2], df_null,          "Null (shuffled perm)",    n_null)

    # save
    fname = roi.replace('/','_').replace('\\','_') + '_6grid.png'
    fig.savefig(os.path.join(out_compare, fname),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
