#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:55:47 2025
@author: Patrick Hackett

End-to-end MiniRocket one-vs-all with permutation shuffle-train,
using re-epoched windows under the “all-or-nothing” trial‐NaN assumption:

  • Fixation:   trialinfo[:,0] ± 750 ms  (Half fix half baseline/iti, ends 250 ms before encoding starts)
  • Encoding:   trialinfo[:,2] → +1500 ms  
  • Delay1:     trialinfo[:,4] + 100 ms → +1500 ms  
  • Delay2:     trialinfo[:,4] +1600 ms → +1500 ms  

Invalid trials (any NaN in trialinfo row) are dropped outright.
"""
import os
import scipy.io
import pandas as pd
import numpy as np
from time import time
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.base import clone
from scipy.stats import binomtest
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ── USER PARAMETERS ───────────────────────────────────────────────────────────
DATA_DIR          = r'E:\data\project_repos\phzhr_turtles_av_ml\data'
MODEL_RESULTS_DIR = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
FILE_SUFFIX       = '_uV_allregions_iti.mat'
n_cores           = 32          # number of parallel workers
n_perm            = 100        # permutations per channel/class
alpha             = 0.05        # significance threshold
window_len        = 1500        # ms per window
np.random.seed(42)               # reproducibility
# ──────────────────────────────────────────────────────────────────────────────

# 1) Find all subject MAT files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

# 2) Build region→subjects map & keep regions with ≥3 subjects
region_subjects = defaultdict(set)
for fname in mat_files:
    mat  = scipy.io.loadmat(os.path.join(DATA_DIR, fname))
    labs = mat['anatomicallabels']
    subj = fname.split('_')[0]
    for row in labs:
        lbl = row[1]
        if lbl is None or np.size(lbl) == 0:
            continue
        txt = (lbl[0].strip() if isinstance(lbl, (list, np.ndarray))
               else str(lbl).strip())
        if txt and txt != 'Unknown':
            region_subjects[txt].add(subj)

df_regs = (
    pd.DataFrame([
        {'Region': r, 'SubjectCount': len(s), 'Subjects': sorted(s)}
        for r, s in region_subjects.items()
    ])
    .query("SubjectCount >= 3")
    .reset_index(drop=True)
)
included_regions = set(df_regs['Region'])
top_group        = sorted({sub for subs in df_regs['Subjects'] for sub in subs})

# ── Helper functions ─────────────────────────────────────────────────────────
def get_region_labels(mat_labels):
    """Extract clean ROI names from MATLAB anatomical labels."""
    out = []
    for c in mat_labels[:,1]:
        if c is None or np.size(c) == 0:
            out.append(None)
        else:
            v = (c.item() if isinstance(c, np.ndarray) and c.size == 1
                 else c[0] if isinstance(c, list)
                 else str(c))
            v = v.strip()
            out.append(v if v != 'Unknown' else None)
    return out

def get_valid_trials(tri, n_samps):
    """
    A trial is valid iff *all* its trialinfo entries
    tri[t,0], tri[t,2], tri[t,4], tri[t,6] are non-NaN,
    and all four windows fit within [0, n_samps).
    """
    valid = []
    for t in range(tri.shape[0]):
        if np.isnan(tri[t, [0,2,4,6]]).any():
            continue
        t0, t2, t4 = map(int, (tri[t,0], tri[t,2], tri[t,4]))
        if (t0 - 750 < 0
            or t0 + 750 > n_samps
            or t2 + window_len > n_samps
            or t4 + 100 + 2 * window_len > n_samps):
            continue
        valid.append(t)
    return valid

def extract_window(data, tri, trials, ch, sc, length, offset=0):
    """
    Slice data[t, ch, start:start+length] for each t in trials,
    where start = tri[t, sc] + offset. Assumes all-or-nothing NaNs.
    Returns X shape (n_trials,1,length), and y from tri[t,6].
    """
    X, y = [], []
    n_samps = data.shape[2]
    for t in trials:
        st = int(tri[t, sc]) + offset
        if st < 0 or st + length > n_samps:
            continue
        X.append(data[t, ch, st:st+length])
        y.append(int(tri[t,6]))
    if not X:
        return np.empty((0,1,length)), np.empty((0,), int)
    return np.stack(X)[:, None, :], np.array(y, int)

def process_channel(subject, data, tri, region_labels, valid_trials, ch):
    """
    One-vs-all MiniRocket + Ridge with permutation shuffle-train,
    but *train* on Delay1 instead of Encoding.

    Windows (all length=window_len):
      • Fixation: trialinfo[:,0] ±750ms  
      • Encoding: trialinfo[:,2] → +1500ms  
      • Delay1:   trialinfo[:,4] +100ms → +1500ms  ← TRAIN ON THIS
      • Delay2:   trialinfo[:,4] +1600ms → +1500ms  
    """
    out = []
    try:
        # 1) extract the four windows exactly as before
        X_fix, y_fix = extract_window(data, tri, valid_trials, ch,
                                      sc=0, length=window_len, offset=-750)
        X_enc, y_enc = extract_window(data, tri, valid_trials, ch,
                                      sc=2, length=window_len, offset=0)
        X_d1,  y_d1  = extract_window(data, tri, valid_trials, ch,
                                      sc=4, length=window_len, offset=100)
        X_d2,  y_d2  = extract_window(data, tri, valid_trials, ch,
                                      sc=4, length=window_len,
                                      offset=100+window_len)

        # 2) make sure we have enough Delay1 trials (and fix)
        if (len(y_d1) < 5 or len(set(y_d1)) < 2
            or len(y_fix) < 5):
            return out

        # 3) one-vs-all on the Delay1 labels, not Encoding
        for L in np.unique(y_d1):
            pos = np.where(y_d1 == L)[0]
            neg = np.where(y_d1 != L)[0]
            if len(pos) < 5 or len(neg) < 5:
                continue

            # 3a) down-sample negatives to match positives
            try:
                neg_s = train_test_split(
                    neg,
                    train_size=len(pos),
                    stratify=y_d1[neg],
                    random_state=42
                )[0]
            except ValueError:
                neg_s = np.random.choice(neg, size=len(pos), replace=False)

            keep = np.hstack((pos, neg_s))

            # ← **CHANGE**: train on Delay1 window
            Xb, yb = X_d1[keep], (y_d1[keep] == L).astype(int)

            # 4) fit miniROCKET on Delay1 training epochs
            rk = MiniRocket(num_kernels=10000, random_state=42)
            rk.fit(Xb)

            # 5) transform all windows once
            feat_b   = rk.transform(Xb)
            feat_fix = rk.transform(X_fix)
            feat_enc = rk.transform(X_enc)
            feat_d1  = rk.transform(X_d1)
            feat_d2  = rk.transform(X_d2)

            # 6) train Ridge on the Delay1 features
            clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
            clf.fit(feat_b, yb)

            # 7) compute true accuracies on *all* windows
            a_fx = accuracy_score((y_fix==L).astype(int), clf.predict(feat_fix))
            a_en = accuracy_score((y_enc==L).astype(int), clf.predict(feat_enc))
            a_d1 = accuracy_score((y_d1==L).astype(int),   clf.predict(feat_d1))
            a_d2 = accuracy_score((y_d2==L).astype(int),   clf.predict(feat_d2))

            # 8) build permutation nulls (shuffle only the Delay1 train labels)
            null_fx = np.zeros(n_perm)
            null_en = np.zeros(n_perm)
            null_d1 = np.zeros(n_perm)
            null_d2 = np.zeros(n_perm)
            for i in range(n_perm):
                yb_sh = np.random.permutation(yb)
                perm  = clone(clf)
                perm.fit(feat_b, yb_sh)
                null_fx[i] = accuracy_score((y_fix==L).astype(int),
                                            perm.predict(feat_fix))
                null_en[i] = accuracy_score((y_enc==L).astype(int),
                                            perm.predict(feat_enc))
                null_d1[i] = accuracy_score((y_d1==L).astype(int),
                                            perm.predict(feat_d1))
                null_d2[i] = accuracy_score((y_d2==L).astype(int),
                                            perm.predict(feat_d2))

            # 9) compute null means/stds and p-values
            fx_nm, fx_ns = null_fx.mean(), null_fx.std(ddof=1)
            en_nm, en_ns = null_en.mean(), null_en.std(ddof=1)
            d1_nm,d1_ns  = null_d1.mean(), null_d1.std(ddof=1)
            d2_nm,d2_ns  = null_d2.mean(), null_d2.std(ddof=1)

            p_fx_perm = (null_fx >= a_fx).sum()+1
            p_en_perm = (null_en >= a_en).sum()+1
            p_d1_perm = (null_d1 >= a_d1).sum()+1
            p_d2_perm = (null_d2 >= a_d2).sum()+1
            p_fx_perm /= (n_perm+1)
            p_en_perm /= (n_perm+1)
            p_d1_perm /= (n_perm+1)
            p_d2_perm /= (n_perm+1)

            # 10) paired Delay vs Fix tests
            obs_d1   = a_d1 - a_fx
            obs_d2   = a_d2 - a_fx
            null_pd1 = null_d1 - null_fx
            null_pd2 = null_d2 - null_fx
            p_d1fx = ((null_pd1 >= obs_d1).sum()+1)/(n_perm+1)
            p_d2fx = ((null_pd2 >= obs_d2).sum()+1)/(n_perm+1)

            # 11) exact binomial tests
            def btest(pred, true):
                k = (pred==true).sum()
                return binomtest(k, len(true), p=0.5,
                                 alternative='greater').pvalue

            p_fx_bin = btest(clf.predict(feat_fix), (y_fix==L).astype(int))
            p_en_bin = btest(clf.predict(feat_enc), (y_enc==L).astype(int))
            p_d1_bin = btest(clf.predict(feat_d1),  (y_d1==L).astype(int))
            p_d2_bin = btest(clf.predict(feat_d2),  (y_d2==L).astype(int))

            # 12) record everything exactly as before
            out.append({
                'Subject':          subject,
                'ChannelIndex':     ch,
                'ROI':              region_labels[ch],
                'ClassLabel':       int(L),
                'FixationAccuracy': round(a_fx,3),
                'EncodingAccuracy': round(a_en,3),
                'Delay1Accuracy':   round(a_d1,3),
                'Delay2Accuracy':   round(a_d2,3),
                'Fix_null_mean':    round(fx_nm,3),
                'Fix_null_std':     round(fx_ns,3),
                'Enc_null_mean':    round(en_nm,3),
                'Enc_null_std':     round(en_ns,3),
                'D1_null_mean':     round(d1_nm,3),
                'D1_null_std':      round(d1_ns,3),
                'D2_null_mean':     round(d2_nm,3),
                'D2_null_std':      round(d2_ns,3),
                'Fix_p_perm':       round(p_fx_perm,4),
                'Enc_p_perm':       round(p_en_perm,4),
                'D1_p_perm':        round(p_d1_perm,4),
                'D2_p_perm':        round(p_d2_perm,4),
                'D1_vsFix_p':       round(p_d1fx,4),
                'D2_vsFix_p':       round(p_d2fx,4),
                'Fix_p_binom':      round(p_fx_bin,4),
                'Enc_p_binom':      round(p_en_bin,4),
                'D1_p_binom':       round(p_d1_bin,4),
                'D2_p_binom':       round(p_d2_bin,4),
                'Fix_signif_binom': p_fx_bin < alpha,
                'Enc_signif_binom': p_en_bin < alpha,
                'D1_signif_binom':  p_d1_bin < alpha,
                'D2_signif_binom':  p_d2_bin < alpha,
                'D1_signif_vsFix':  p_d1fx  < alpha,
                'D2_signif_vsFix':  p_d2fx  < alpha,
            })

    except Exception as e:
        print(f"⚠ Subj {subject}, Ch {ch} error: {e}")

    return out
# ── MAIN LOOP ────────────────────────────────────────────────────────────────
results = []
print("Starting parallel MiniRocket…")
t0 = time()

for idx, subject in enumerate(top_group, start=1):
    matf = next((f for f in mat_files if f.startswith(subject)), None)
    if matf is None:
        continue
    mat           = scipy.io.loadmat(os.path.join(DATA_DIR, matf))
    data, tri     = mat['data_mat'], mat['trialinfo']
    region_labels = get_region_labels(mat['anatomicallabels'])
    valid_trials  = get_valid_trials(tri, data.shape[2])
    if len(valid_trials) < 10:
        print(f"[{idx}/{len(top_group)}] {subject}: only {len(valid_trials)} valid trials, skipping")
        continue

    roi_chs = [i for i,r in enumerate(region_labels) if r in included_regions]
    print(f"[{idx}/{len(top_group)}] {subject}: {len(roi_chs)} channels")
    with tqdm_joblib(tqdm(desc=subject, total=len(roi_chs), ncols=80)):
        chunks = Parallel(n_jobs=n_cores)(
            delayed(process_channel)(
                subject, data, tri, region_labels, valid_trials, ch
            ) for ch in roi_chs
        )
    for sublist in chunks:
        results.extend(sublist)

print("Finished in", round(time()-t0,1), "s — models:", len(results))

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
outdir = os.path.join(MODEL_RESULTS_DIR, '13_permutationshuffledtrain_iti')
os.makedirs(outdir, exist_ok=True)
pd.DataFrame(results).to_csv(
    os.path.join(outdir, '13_onevsall_with_permutationontrain_iti_dptrain.csv'),
    index=False
)
