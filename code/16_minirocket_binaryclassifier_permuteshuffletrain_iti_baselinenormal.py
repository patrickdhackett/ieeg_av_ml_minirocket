#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end MiniRocket one-vs-all with permutation shuffle-train,
using per-trial baseline normalization on the middle 1 000 ms of ITI/fixation
to remove trial‐specific offsets/drifts.
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
n_cores           = 48      # parallel workers
n_perm            = 100    # permutations per channel/class
alpha             = 0.05    # significance threshold
window_len        = 1500    # ms per window
np.random.seed(42)          # reproducibility

# baseline‐normalization window (absolute samples, since sfreq=1 kHz)
BASE_OFF = 500   # start of the middle 1000 ms of ITI
BASE_LEN = 1000  # length in samples

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_region_labels(mat_labels):
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
    valid = []
    for t in range(tri.shape[0]):
        if np.isnan(tri[t, [0,2,4,6]]).any():
            continue
        t0, t2, t4 = map(int, (tri[t,0], tri[t,2], tri[t,4]))
        # check that all windows fit
        if (t0 - 750 < 0
            or t0 + 750 > n_samps
            or t2 + window_len > n_samps
            or t4 + 100 + 2*window_len > n_samps):
            continue
        valid.append(t)
    return valid

def extract_window(data, tri, trials, ch, sc, length, offset=0):
    """
    Slice data[t, ch, start:start+length] for each t in trials,
    where start = tri[t, sc] + offset.  Returns X (n_epochs, 1, length)
    and y (n_epochs,) from tri[:,6].
    Assumes `data` has already been baseline‐normalized.
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
    out = []
    # extract all four windows (fix, enc, d1, d2)
    X_fix, y_fix = extract_window(data, tri, valid_trials, ch, sc=0, length=window_len, offset=-750)
    X_enc, y_enc = extract_window(data, tri, valid_trials, ch, sc=2, length=window_len, offset=0)
    X_d1,  y_d1  = extract_window(data, tri, valid_trials, ch, sc=4, length=window_len, offset=100)
    X_d2,  y_d2  = extract_window(data, tri, valid_trials, ch, sc=4,
                                   length=window_len, offset=100+window_len)

    # require at least 5 trials per condition
    if (len(y_enc) < 5 or len(set(y_enc)) < 2
        or len(y_fix) < 5 or len(y_d1) < 5 or len(y_d2) < 5):
        return out

    for L in np.unique(y_enc):
        pos = np.where(y_enc == L)[0]
        neg = np.where(y_enc != L)[0]
        if len(pos) < 5 or len(neg) < 5:
            continue

        # downsample negatives
        try:
            neg_s = train_test_split(
                neg, train_size=len(pos),
                stratify=y_enc[neg], random_state=42
            )[0]
        except ValueError:
            neg_s = np.random.choice(neg, size=len(pos), replace=False)

        keep = np.hstack((pos, neg_s))
        Xb, yb = X_enc[keep], (y_enc[keep] == L).astype(int)

        # fit MiniRocket once
        rk = MiniRocket(num_kernels=10000, random_state=42)
        rk.fit(Xb)
        feat_b   = rk.transform(Xb)
        feat_fix = rk.transform(X_fix)
        feat_enc = rk.transform(X_enc)
        feat_d1  = rk.transform(X_d1)
        feat_d2  = rk.transform(X_d2)

        # train & eval Ridge
        clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
        clf.fit(feat_b, yb)

        a_fx = accuracy_score((y_fix==L).astype(int), clf.predict(feat_fix))
        a_en = accuracy_score((y_enc==L).astype(int), clf.predict(feat_enc))
        a_d1 = accuracy_score((y_d1==L).astype(int),   clf.predict(feat_d1))
        a_d2 = accuracy_score((y_d2==L).astype(int),   clf.predict(feat_d2))

        # permutation nulls
        null_fx = np.zeros(n_perm)
        null_en = np.zeros(n_perm)
        null_d1 = np.zeros(n_perm)
        null_d2 = np.zeros(n_perm)
        for i in range(n_perm):
            yb_sh = np.random.permutation(yb)
            perm  = clone(clf)
            perm.fit(feat_b, yb_sh)
            null_fx[i] = accuracy_score((y_fix==L).astype(int), perm.predict(feat_fix))
            null_en[i] = accuracy_score((y_enc==L).astype(int), perm.predict(feat_enc))
            null_d1[i] = accuracy_score((y_d1==L).astype(int), perm.predict(feat_d1))
            null_d2[i] = accuracy_score((y_d2==L).astype(int), perm.predict(feat_d2))

        # summarize
        fx_nm, fx_ns = null_fx.mean(), null_fx.std(ddof=1)
        en_nm, en_ns = null_en.mean(), null_en.std(ddof=1)
        d1_nm,d1_ns  = null_d1.mean(), null_d1.std(ddof=1)
        d2_nm,d2_ns  = null_d2.mean(), null_d2.std(ddof=1)

        p_fx_perm = ((null_fx >= a_fx).sum()+1)/(n_perm+1)
        p_en_perm = ((null_en >= a_en).sum()+1)/(n_perm+1)
        p_d1_perm = ((null_d1 >= a_d1).sum()+1)/(n_perm+1)
        p_d2_perm = ((null_d2 >= a_d2).sum()+1)/(n_perm+1)

        # paired tests
        obs_d1   = a_d1 - a_fx
        null_pd1 = null_d1 - null_fx
        p_d1fx   = ((null_pd1 >= obs_d1).sum()+1)/(n_perm+1)

        obs_d2   = a_d2 - a_fx
        null_pd2 = null_d2 - null_fx
        p_d2fx   = ((null_pd2 >= obs_d2).sum()+1)/(n_perm+1)

        # binomial
        def btest(pred,true):
            k = (pred==true).sum()
            return binomtest(k, len(true), p=0.5,
                             alternative='greater').pvalue

        p_fx_bin = btest(clf.predict(feat_fix), (y_fix==L).astype(int))
        p_en_bin = btest(clf.predict(feat_enc), (y_enc==L).astype(int))
        p_d1_bin = btest(clf.predict(feat_d1),  (y_d1==L).astype(int))
        p_d2_bin = btest(clf.predict(feat_d2),  (y_d2==L).astype(int))

        out.append({
            'Subject':         subject,
            'ChannelIndex':    ch,
            'ROI':             region_labels[ch],
            'ClassLabel':      int(L),
            'FixationAccuracy':a_fx,
            'EncodingAccuracy':a_en,
            'Delay1Accuracy':  a_d1,
            'Delay2Accuracy':  a_d2,
            'Fix_null_mean':   fx_nm,
            'Fix_null_std':    fx_ns,
            'Enc_null_mean':   en_nm,
            'Enc_null_std':    en_ns,
            'D1_null_mean':    d1_nm,
            'D1_null_std':     d1_ns,
            'D2_null_mean':    d2_nm,
            'D2_null_std':     d2_ns,
            'Fix_p_perm':      p_fx_perm,
            'Enc_p_perm':      p_en_perm,
            'D1_p_perm':       p_d1_perm,
            'D2_p_perm':       p_d2_perm,
            'D1_vsFix_p':      p_d1fx,
            'D2_vsFix_p':      p_d2fx,
            'Fix_p_binom':     p_fx_bin,
            'Enc_p_binom':     p_en_bin,
            'D1_p_binom':      p_d1_bin,
            'D2_p_binom':      p_d2_bin,
            'Fix_signif_binom':p_fx_bin < alpha,
            'Enc_signif_binom':p_en_bin < alpha,
            'D1_signif_binom': p_d1_bin < alpha,
            'D2_signif_binom': p_d2_bin < alpha,
            'D1_signif_vsFix': p_d1fx   < alpha,
            'D2_signif_vsFix': p_d2fx   < alpha,
        })

    return out

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
results = []
print("Starting parallel MiniRocket…")
t0 = time()

for subj in tqdm(sorted({s for s in os.listdir(DATA_DIR) if s.endswith(FILE_SUFFIX)}), desc="Subjects"):
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, subj))
    data, tri = mat['data_mat'], mat['trialinfo']
    region_labels = get_region_labels(mat['anatomicallabels'])
    n_samps = data.shape[2]

    # 1) compute per‐trial/channel baseline and normalize
    baseline = data[:,:,BASE_OFF:BASE_OFF+BASE_LEN]
    mean0    = baseline.mean(axis=2, keepdims=True)     # shape (n_trials,n_chans,1)
    std0     = baseline.std(axis=2, ddof=1, keepdims=True)
    data     = (data - mean0) / (std0 + 1e-6)

    # 2) pick valid trials
    valid = get_valid_trials(tri, n_samps)
    if len(valid) < 10:
        continue

    # 3) process each channel in parallel
    roi_chs = [i for i,r in enumerate(region_labels) if r is not None]
    with tqdm_joblib(tqdm(desc=subj, total=len(roi_chs))) as pbar:
        chunks = Parallel(n_jobs=n_cores)(
            delayed(process_channel)(
                subj, data, tri, region_labels, valid, ch
            ) for ch in roi_chs
        )

    for out in chunks:
        results.extend(out)

print("Done in", round(time()-t0,1), "s — saved models:", len(results))

# ── SAVE TO CSV ─────────────────────────────────────────────────────────────
outdir = os.path.join(MODEL_RESULTS_DIR, '16_timefreq_baselinenorm')
os.makedirs(outdir, exist_ok=True)
pd.DataFrame(results).to_csv(
    os.path.join(outdir, '16_onevsall_mininirocket_baselinenorm.csv'),
    index=False
)
