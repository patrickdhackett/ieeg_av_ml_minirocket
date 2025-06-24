#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:55:47 2025
@author: a1_phzhr

– No permutation-null control
– Three 1.5 s delay windows
– Binomial + McNemar tests
"""
import os
import scipy.io
import pandas as pd
import numpy as np
from time import time
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket
from scipy.stats import binomtest
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ── USER PARAMETERS ───────────────────────────────────────────────────────────
DATA_DIR          = r'E:\data\project_repos\phzhr_turtles_av_ml\data'
MODEL_RESULTS_DIR = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
FILE_SUFFIX       = '_uV_allregions.mat'
n_cores           = 32           # parallel workers
alpha             = 0.05         # sig threshold
window_len        = 1500         # ms per slice
max_off           = 4000         # ensure full 4 s delay fits
np.random.seed(42)               # reproducibility
# ──────────────────────────────────────────────────────────────────────────────

# 1) Gather .mat files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

# 2) Region→subjects map & keep ROIs with ≥3 subjects
region_subjects = defaultdict(set)
for fname in mat_files:
    mat   = scipy.io.loadmat(os.path.join(DATA_DIR, fname))
    labs  = mat['anatomicallabels']
    subj  = fname.split('_')[0]
    regs  = set()
    for row in labs:
        lbl = row[1]
        if lbl is None or np.size(lbl)==0:
            continue
        txt = lbl[0].strip() if isinstance(lbl, (list,np.ndarray)) else str(lbl).strip()
        if txt and txt not in ('Unknown',''):
            regs.add(txt)
    for r in regs:
        region_subjects[r].add(subj)

df_regs = (
    pd.DataFrame([{'Region':r,'SubjectCount':len(s),'Subjects':sorted(s)}
                  for r,s in region_subjects.items()])
    .query("SubjectCount>=3")
    .reset_index(drop=True)
)
included_regions = set(df_regs['Region'])
top_group        = sorted({sub for subs in df_regs['Subjects'] for sub in subs})

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_region_labels(labels):
    out=[]
    for c in labels[:,1]:
        if c is None or np.size(c)==0:
            out.append(None)
        else:
            v = c.item() if isinstance(c,np.ndarray) and c.size==1 \
                else c[0] if isinstance(c,list) else str(c)
            v=v.strip()
            out.append(v if v not in ('Unknown','') else None)
    return out

def get_valid_trials(tri, n_samples, max_off):
    return [
        t for t in range(tri.shape[0])
        if (not np.isnan(tri[t,0]) and not np.isnan(tri[t,6])
            and int(tri[t,0]) + max_off <= n_samples
            and int(tri[t,3]) + 50 + max_off <= n_samples)
    ]

def extract_window(data, tri, trials, ch, sc, length, offset=0):
    X,y = [],[]
    for t in trials:
        st  = int(tri[t,sc]) + offset
        win = data[t,ch,st:st+length]
        if not np.isnan(win).any():
            X.append(win); y.append(int(tri[t,6]))
    if X:
        return np.stack(X)[:,None,:], np.array(y)
    return np.empty((0,1,length)), np.empty((0,),int)

# ── Core per-channel worker ───────────────────────────────────────────────────
def process_channel(subject,data,tri,region_labels,valid_trials,ch):
    out=[]
    try:
        #  — training set from encoding (sc=0) —
        X_enc, y_enc = extract_window(data,tri,valid_trials,ch,sc=0,length=window_len)
        if len(y_enc)<5 or len(set(y_enc))<2:
            return out

        for L in np.unique(y_enc):
            pos = np.where(y_enc==L)[0]
            neg = np.where(y_enc!=L)[0]
            if len(pos)<5 or len(neg)<5:
                continue

            # balanced sampling
            try:
                neg_s = train_test_split(
                    neg, train_size=len(pos),
                    stratify=y_enc[neg], random_state=42
                )[0]
            except ValueError:
                neg_s = np.random.choice(neg, size=len(pos), replace=False)

            keep = np.hstack([pos,neg_s])
            Xb, yb = X_enc[keep], (y_enc[keep]==L).astype(int)

            # pipeline
            pipe = make_pipeline(
                MiniRocket(num_kernels=10000, random_state=42),
                RidgeClassifierCV(alphas=np.logspace(-3,3,10))
            )
            pipe.fit(Xb,yb)

            # held-out fixation
            X_fix, y_fix = [],[]
            for t in valid_trials:
                raw = data[t,ch,:1000]
                if np.isnan(raw).any(): continue
                pad = np.zeros(window_len); pad[:1000]=raw
                X_fix.append(pad); y_fix.append(int(tri[t,6]))
            X_fix = np.stack(X_fix)[:,None,:] if X_fix else np.empty((0,1,window_len))
            y_fix = np.array(y_fix)         if y_fix else np.empty((0,),int)

            # held-out encoding = X_enc,y_enc
            X_hold_enc, y_hold_enc = X_enc, y_enc

            # three delay slices
            X_dp1,y_dp1 = extract_window(data,tri,valid_trials,ch,sc=3,length=window_len,offset=0)
            X_dp2,y_dp2 = extract_window(data,tri,valid_trials,ch,sc=3,length=window_len,offset=1500)
            X_dp3,y_dp3 = extract_window(data,tri,valid_trials,ch,sc=3,length=window_len,offset=2500)

            # predictions
            y_fix_pred = pipe.predict(X_fix)
            y_enc_pred = pipe.predict(X_hold_enc)
            y_dp1_pred = pipe.predict(X_dp1)
            y_dp2_pred = pipe.predict(X_dp2)
            y_dp3_pred = pipe.predict(X_dp3)

            # accuracies
            acc_fx  = accuracy_score((y_fix == L).astype(int),   y_fix_pred)
            acc_en  = accuracy_score((y_hold_enc==L).astype(int),y_enc_pred)
            acc_dp1 = accuracy_score((y_dp1  ==L).astype(int),  y_dp1_pred)
            acc_dp2 = accuracy_score((y_dp2  ==L).astype(int),  y_dp2_pred)
            acc_dp3 = accuracy_score((y_dp3  ==L).astype(int),  y_dp3_pred)

            # binomial tests
            p_fx  = binomtest((y_fix_pred== (y_fix==L)).sum(),   len(y_fix),   p=0.5,alternative='greater').pvalue
            p_en  = binomtest((y_enc_pred== (y_hold_enc==L)).sum(),len(y_hold_enc),p=0.5,alternative='greater').pvalue
            p_dp1 = binomtest((y_dp1_pred==(y_dp1==L)).sum(),   len(y_dp1),   p=0.5,alternative='greater').pvalue
            p_dp2 = binomtest((y_dp2_pred==(y_dp2==L)).sum(),   len(y_dp2),   p=0.5,alternative='greater').pvalue
            p_dp3 = binomtest((y_dp3_pred==(y_dp3==L)).sum(),   len(y_dp3),   p=0.5,alternative='greater').pvalue

            # McNemar helper
            def mcnemar(y_t, p1, p2):
                c1 = (p1==y_t); c2=(p2==y_t)
                n10 = np.sum(c1 & ~c2); n01=np.sum(~c1 & c2)
                return np.nan if n10+n01==0 else \
                    binomtest(n01, n10+n01, p=0.5, alternative='greater').pvalue

            p_d1_fx = mcnemar(y_fix, y_fix_pred, y_dp1_pred)
            p_d2_fx = mcnemar(y_fix, y_fix_pred, y_dp2_pred)
            p_d3_fx = mcnemar(y_fix, y_fix_pred, y_dp3_pred)

            out.append({
                'Subject':            subject,
                'ChannelIndex':       ch,
                'ROI':                region_labels[ch],
                'ClassLabel':         int(L),
                'FixationAccuracy':   round(acc_fx,3),
                'EncodingAccuracy':   round(acc_en,3),
                'Delay1Accuracy':     round(acc_dp1,3),
                'Delay2Accuracy':     round(acc_dp2,3),
                'Delay3Accuracy':     round(acc_dp3,3),
                'Fix_p_binom':        round(p_fx,4),
                'Enc_p_binom':        round(p_en,4),
                'Delay1_p_binom':     round(p_dp1,4),
                'Delay2_p_binom':     round(p_dp2,4),
                'Delay3_p_binom':     round(p_dp3,4),
                'Delay1_vsFix_p':     None if np.isnan(p_d1_fx) else round(p_d1_fx,4),
                'Delay2_vsFix_p':     None if np.isnan(p_d2_fx) else round(p_d2_fx,4),
                'Delay3_vsFix_p':     None if np.isnan(p_d3_fx) else round(p_d3_fx,4),
                'Fix_signif_binom':   p_fx  < alpha,
                'Enc_signif_binom':   p_en  < alpha,
                'Delay1_signif_binom':p_dp1 < alpha,
                'Delay2_signif_binom':p_dp2 < alpha,
                'Delay3_signif_binom':p_dp3 < alpha,
            })

    except Exception as e:
        print(f"⚠ Subj {subject}, Ch {ch} error: {e}")
    return out

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
results = []
print("Starting MiniRocket (no perm null)…")
t0 = time()

for idx, subject in enumerate(top_group, start=1):
    # load subject mat
    matf = next((f for f in mat_files if f.startswith(subject)), None)
    if not matf:
        continue
    mat     = scipy.io.loadmat(os.path.join(DATA_DIR, matf))
    data    = mat['data_mat']
    tri     = mat['trialinfo']
    labels  = mat['anatomicallabels']
    region_labels = get_region_labels(labels)
    valid_trials  = get_valid_trials(tri, data.shape[2], max_off)
    if len(valid_trials) < 10:
        continue

    roi_chs = [i for i,r in enumerate(region_labels) if r in included_regions]
    with tqdm_joblib(tqdm(f"{subject}", total=len(roi_chs), ncols=80)):
        chunk = Parallel(n_jobs=n_cores)(
            delayed(process_channel)(
                subject, data, tri, region_labels, valid_trials, ch
            ) for ch in roi_chs
        )
    for sublist in chunk:
        results.extend(sublist)

print("Done in", round(time()-t0,1), "s — models:", len(results))

# ── SAVE ─────────────────────────────────────────────────────────────────────
outdir = os.path.join(MODEL_RESULTS_DIR, '12_no_permutation')
os.makedirs(outdir, exist_ok=True)
pd.DataFrame(results).to_csv(
    os.path.join(outdir, '12_onevsall_noPerm_with_multipleDelays.csv'),
    index=False
)
