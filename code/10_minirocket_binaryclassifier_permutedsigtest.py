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

# ── USER PARAMETERS ───────────────────────────────────────────────────────────
DATA_DIR          = r'E:\data\project_repos\phzhr_turtles_av_ml\data'
MODEL_RESULTS_DIR = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
FILE_SUFFIX       = '_uV_allregions.mat'
n_cores           = 32          # how many parallel workers
n_perm            = 1000        # number of permutations
alpha             = 0.05        # significance threshold
np.random.seed(42)               # reproducibility
# ──────────────────────────────────────────────────────────────────────────────

# 1) Gather .mat files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

# 2) Build region→subjects map & pick ROIs with ≥3 subjects
region_subjects = defaultdict(set)
for fname in mat_files:
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, fname))
    labs = mat['anatomicallabels']
    subj = fname.split('_')[0]
    regs = set()
    for row in labs:
        lbl = row[1]
        if lbl is None or np.size(lbl) == 0:
            continue
        txt = lbl[0].strip() if isinstance(lbl, (list, np.ndarray)) else str(lbl).strip()
        if txt and txt not in ('Unknown',''):
            regs.add(txt)
    for r in regs:
        region_subjects[r].add(subj)

df_regs = (
    pd.DataFrame([
        {'Region': r, 'SubjectCount': len(s), 'Subjects': sorted(s)}
        for r, s in region_subjects.items()
    ])
    .query("SubjectCount >= 3")
    .reset_index(drop=True)
)
included_regions = set(df_regs['Region'])
top_group = sorted({s for subs in df_regs['Subjects'] for s in subs})

# ── Helper functions ──────────────────────────────────────────────────────────
def get_region_labels(labels):
    out = []
    for c in labels[:,1]:
        if c is None or np.size(c) == 0:
            out.append(None)
        else:
            v = c.item() if isinstance(c, np.ndarray) and c.size == 1 \
                else c[0] if isinstance(c, list) else str(c)
            v = v.strip()
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
    X, y = [], []
    for t in trials:
        st = int(tri[t, sc]) + offset
        win = data[t, ch, st:st+length]
        if not np.isnan(win).any():
            X.append(win)
            y.append(int(tri[t,6]))
    if X:
        return np.stack(X)[:, None, :], np.array(y)
    return np.empty((0,1,length)), np.empty((0,), int)

# Pre‐instantiate MiniRocket so kernels are shared
rocket = MiniRocket(num_kernels=10000, random_state=42)

def process_channel(subject, data, tri, region_labels, valid_trials, ch, window_len):
    out = []
    try:
        # — extract windows —
        X_tr, y_tr           = extract_window(data, tri, valid_trials, ch, 0, window_len)
        X_val_enc, y_val_enc = X_tr, y_tr
        X_val_dp, y_val_dp   = extract_window(data, tri, valid_trials, ch, 3, window_len)
        # fixation (pad to window_len)
        X_fix, y_fix = [], []
        for t in valid_trials:
            raw = data[t, ch, :1000]
            if np.isnan(raw).any(): 
                continue
            pad = np.zeros(window_len)
            pad[:1000] = raw
            X_fix.append(pad)
            y_fix.append(int(tri[t,6]))
        X_val_fix = np.stack(X_fix)[:, None, :] if X_fix else np.empty((0,1,window_len))
        y_val_fix = np.array(y_fix)         if y_fix else np.empty((0,), int)

        # — skip if too few trials —
        if (len(y_tr) < 5 or len(set(y_tr)) < 2
            or len(y_val_enc) < 5 or len(y_val_dp) < 5 or len(y_val_fix) < 5):
            return out

        # — one‐vs‐all loop —
        for L in np.unique(y_tr):
            pos = np.where(y_tr == L)[0]
            neg = np.where(y_tr != L)[0]
            if len(pos) < 5 or len(neg) < 5:
                continue
            # stratified downsample negatives
            try:
                neg_s = train_test_split(
                    neg, train_size=len(pos),
                    stratify=y_tr[neg], random_state=42
                )[0]
            except ValueError:
                neg_s = np.random.choice(neg, size=len(pos), replace=False)

            keep = np.hstack([pos, neg_s])
            Xb, yb = X_tr[keep], (y_tr[keep] == L).astype(int)

            # train minirocket→ridge
            pipe = make_pipeline(rocket, RidgeClassifierCV(alphas=np.logspace(-3,3,10)))
            pipe.fit(Xb, yb)

            # single‐pass predictions
            y_pred_fix = pipe.predict(X_val_fix)
            y_pred_enc = pipe.predict(X_val_enc)
            y_pred_dp  = pipe.predict(X_val_dp)

            # observed accuracies
            acc_fx = accuracy_score((y_val_fix == L).astype(int), y_pred_fix)
            acc_en = accuracy_score((y_val_enc == L).astype(int), y_pred_enc)
            acc_dp = accuracy_score((y_val_dp  == L).astype(int), y_pred_dp)

            # build permutation nulls
            null_fx = np.zeros(n_perm)
            null_en = np.zeros(n_perm)
            null_dp = np.zeros(n_perm)
            for i in range(n_perm):
                null_fx[i] = np.mean(y_pred_fix == np.random.permutation((y_val_fix == L).astype(int)))
                null_en[i] = np.mean(y_pred_enc == np.random.permutation((y_val_enc == L).astype(int)))
                null_dp[i] = np.mean(y_pred_dp  == np.random.permutation((y_val_dp  == L).astype(int)))

            # null means & stds
            fx_null_mean, fx_null_std = null_fx.mean(), null_fx.std(ddof=1)
            en_null_mean, en_null_std = null_en.mean(), null_en.std(ddof=1)
            dp_null_mean, dp_null_std = null_dp.mean(), null_dp.std(ddof=1)

            # permutation p‐values
            p_fx_perm = (null_fx >= acc_fx).sum() + 1
            p_fx_perm /= (n_perm + 1)
            p_en_perm = (null_en >= acc_en).sum() + 1
            p_en_perm /= (n_perm + 1)
            p_dp_perm = (null_dp >= acc_dp).sum() + 1
            p_dp_perm /= (n_perm + 1)

            # DELAY vs FIXATION paired test
            obs_diff    = acc_dp - acc_fx
            null_diff   = null_dp - null_fx
            p_dpfx_perm = (null_diff >= obs_diff).sum() + 1
            p_dpfx_perm /= (n_perm + 1)
            signif_dpfx = p_dpfx_perm < alpha

            # exact binomial p‐values
            k_fx = (y_pred_fix == (y_val_fix == L).astype(int)).sum()
            p_fx_bin = binomtest(k_fx, len(y_val_fix), p=0.5, alternative='greater').pvalue
            k_en = (y_pred_enc == (y_val_enc == L).astype(int)).sum()
            p_en_bin = binomtest(k_en, len(y_val_enc), p=0.5, alternative='greater').pvalue
            k_dp = (y_pred_dp == (y_val_dp  == L).astype(int)).sum()
            p_dp_bin = binomtest(k_dp, len(y_val_dp),  p=0.5, alternative='greater').pvalue

            # record everything
            out.append({
                'Subject':           subject,
                'ChannelIndex':      ch,
                'ROI':               region_labels[ch],
                'ClassLabel':        L,
                'FixationAccuracy':  round(acc_fx,   3),
                'EncodingAccuracy':  round(acc_en,   3),
                'DelayAccuracy':     round(acc_dp,   3),
                'Fix_null_mean':     round(fx_null_mean,3),
                'Fix_null_std':      round(fx_null_std, 3),
                'Enc_null_mean':     round(en_null_mean,3),
                'Enc_null_std':      round(en_null_std, 3),
                'Delay_null_mean':   round(dp_null_mean,3),
                'Delay_null_std':    round(dp_null_std, 3),
                'Fix_p_perm':        round(p_fx_perm,4),
                'Enc_p_perm':        round(p_en_perm,4),
                'Delay_p_perm':      round(p_dp_perm,4),
                'Obs_DpFx_Diff':     round(obs_diff, 3),
                'DpFx_p_perm':       round(p_dpfx_perm,4),
                'DpFx_signif_perm':  signif_dpfx,
                'Fix_p_binom':       round(p_fx_bin,4),
                'Enc_p_binom':       round(p_en_bin,4),
                'Delay_p_binom':     round(p_dp_bin,4),
                'Fix_signif_perm':   p_fx_perm  < alpha,
                'Enc_signif_perm':   p_en_perm  < alpha,
                'Delay_signif_perm': p_dp_perm  < alpha,
                'Fix_signif_binom':  p_fx_bin   < alpha,
                'Enc_signif_binom':  p_en_bin   < alpha,
                'Delay_signif_binom':p_dp_bin   < alpha
            })

    except Exception as e:
        print(f"⚠ Subj {subject}, Ch {ch} error: {e}")
    return out

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
window_len = 1500
max_off    = window_len + 50
results    = []

print("Starting parallel MiniRocket with paired permutation tests…")
t0 = time()

for idx, subject in enumerate(top_group, start=1):
    matf = next((f for f in mat_files if f.startswith(subject)), None)
    if not matf:
        continue
    print(f"[{idx}/{len(top_group)}] {subject}", flush=True)
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, matf))
    data, tri, labels = mat['data_mat'], mat['trialinfo'], mat['anatomicallabels']
    region_labels    = get_region_labels(labels)
    valid_trials     = get_valid_trials(tri, data.shape[2], max_off)
    if len(valid_trials) < 10:
        continue

    roi_chs = [i for i,r in enumerate(region_labels) if r in included_regions]
    print(f"    → {len(roi_chs)} channels", flush=True)

    channel_results = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(process_channel)(
            subject, data, tri, region_labels,
            valid_trials, ch, window_len
        )
        for ch in tqdm(roi_chs, desc=subject, ncols=80)
    )
    for lst in channel_results:
        results.extend(lst)

print("Finished in", round(time()-t0,1), "s — models:", len(results))

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(MODEL_RESULTS_DIR, '10_permutationsigtest'), exist_ok=True)
pd.DataFrame(results).to_csv(
    os.path.join(MODEL_RESULTS_DIR, '10_permutationsigtest/10_onevsall_with_paired_signif.csv'),
    index=False
)
