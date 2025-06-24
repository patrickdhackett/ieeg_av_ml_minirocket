import os
import scipy.io
import pandas as pd
from collections import defaultdict
import numpy as np
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket


# IMPORT ALL THE DATA FROM THE MAT CELL ARRAYS
DATA_DIR  = r'E:\data\project_repos\phzhr_turtles_av_ml\data'
model_results_dir = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
FILE_SUFFIX = '_uV_allregions.mat'

# Collect matching .mat files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

# Dictionary to store: region  set of subject IDs
region_subjects = defaultdict(set)

# Grab Region names from the labelsAnat
for fname in mat_files:
    fpath = os.path.join(DATA_DIR, fname)
    try:
        mat = scipy.io.loadmat(fpath)
        labels = mat['anatomicallabels']  # shape: (n_channels, 2), dtype=object
        subject_id = fname.split('_')[0]  # Extract subject ID, e.g., "k12wm002"

        # Get all unique region labels (column 1 = index 1)
        regions_in_subject = set(
            row[1][0].strip() if isinstance(row[1], (list, np.ndarray)) else str(row[1]).strip()
            for row in labels
            if row[1] is not None and np.size(row[1]) > 0 and str(row[1]).strip() not in ['Unknown', '']
        )
        
        for region in regions_in_subject:
            region_subjects[region].add(subject_id)

    except Exception as e:
        print(f"L Failed to load or process {fname}: {e}")

# Prepare final output list
rows = []
for region, subject_set in region_subjects.items():
    subject_list = sorted(subject_set)
    rows.append({
        'Region': region,
        'SubjectCount': len(subject_list),
        'Subjects': subject_list
    })

# Convert to DataFrame
df = pd.DataFrame(rows)
df = df.sort_values(by='SubjectCount', ascending=False).reset_index(drop=True)

# Display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

# Show full table
# print(df)


## FILTER REGIONS TO THOSE THAT OVERLAP IN 3 PEOPLE OR MORE, Remove Unknown and empty rows
df_filtered = df[
    (df['SubjectCount'] >= 3) &
    (~df['Region'].isin(['Unknown', '', '[]']))
]

# Define the set of regions to include
included_regions = set(df_filtered['Region'])



print(f"Kept {len(included_regions)} ROIs:\n", sorted(included_regions))

# Get list of all subjects that have any of these regions
top_group = sorted({subj for subjects in df_filtered['Subjects'] for subj in subjects})


### RUN MINIROCKET TARGETED: PER-SUBJECT, PER-CHANNEL MiniRocket with 75/25 train validation split.
## Will use shuffled labels permuted as null to calculate significance


def get_region_labels(labels):
    """Return a list of region names (or None) in channel order."""
    out = []
    for cell in labels[:,1]:
        if cell is None or np.size(cell) == 0:
            out.append(None)
        else:
            # unwrap 1-element arrays or lists
            val = cell.item() if isinstance(cell, np.ndarray) and cell.size==1 else cell[0] if isinstance(cell, list) else str(cell)
            val = val.strip()
            out.append(val if val not in ('Unknown','') else None)
    return out

def get_valid_trials(trialinfo, data_len, max_start_offset):
    """Only keep trials where both encoding & delay windows fit."""
    return [
        t for t in range(trialinfo.shape[0])
        if not (np.isnan(trialinfo[t,0]) or np.isnan(trialinfo[t,6]))
        and (int(trialinfo[t,0]) + max_start_offset <= data_len)
        and (int(trialinfo[t,3]) + 50 + max_start_offset <= data_len)
    ]

def extract_window(data, trialinfo, trials, ch, start_col, length, offset=0):
    """
    Slice out [offset … offset+length) ms from trialinfo[:,start_col].
    Returns X shape (n_trials,1,length) and y shape (n_trials,).
    """
    X, y = [], []
    for t in trials:
        st = int(trialinfo[t, start_col]) + offset
        win = data[t, ch, st:st+length]
        if not np.isnan(win).any():
            X.append(win)
            y.append(int(trialinfo[t,6]))
    if X:
        return np.stack(X)[:, None, :], np.array(y)
    else:
        return np.empty((0,1,length)), np.empty((0,), dtype=int)

# 3) Run per-subject, per-channel MiniRocket

results = []
window_len = 1500
# buffer must cover encoding start + window_len AND delay start + 50 + window_len
max_offset = max(0, 50) + window_len

print("Starting MiniRocket loop…")
t0 = time()

for subject in top_group:
    # load
    subj_file = next((f for f in mat_files if f.startswith(subject)), None)
    if not subj_file:
        print(f"Skipping subject {subject}: file not found.")
        continue
    print(f"Loading data for subject {subject}...")
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, subj_file))
    data = mat['data_mat']
    trialinfo = mat['trialinfo']
    labels = mat['anatomicallabels']
    region_labels = [str(r[1]).strip().strip("[]'") for r in labels]
    roi_chs = [i for i, roi in enumerate(region_labels) if roi in included_regions]
    valid_trials = [
        t for t in range(data.shape[0])
        if not (np.isnan(trialinfo[t,0]) or np.isnan(trialinfo[t,6])) and int(trialinfo[t,3]) + 3100 <= data.shape[2]
    ]
    if len(valid_trials) < 10:
        print(f"  [skip] {subject}: <10 valid trials")
        continue
    
   
    for ch in roi_chs:
        # encoding window → train
        X_train, y_train = extract_window(
            data, trialinfo, valid_trials, ch,
            start_col=0, length=window_len, offset=0
        )
        # delay window → val 
        X_val_dp, y_val_dp = extract_window(
            data, trialinfo, valid_trials, ch,
            start_col=3, length=window_len, offset=0
        )
        # enc window → val 
        X_val_enc, y_val_enc = extract_window(
            data, trialinfo, valid_trials, ch,
            start_col=0, length=window_len, offset=0
        )        
        
        # Pad fix to deal with diff size from encoding. Will need to improve this later with rolling windows
        X_fix_list, y_fix_list = [], []
        for t in valid_trials:
            raw_win = data[t, ch, 0:1000]   # 0…999
            # skip if any NaNs
            if np.isnan(raw_win).any():
                continue
        
            # pad with zeros to length `window_len` (1500)
            pad_size   = window_len - raw_win.shape[0]  # = 500
            padded_win = np.pad(raw_win,
                                (0, pad_size),
                                mode='constant',
                                constant_values=0)
            
            X_fix_list.append(padded_win)
            y_fix_list.append(int(trialinfo[t, 6]))
        
        # now shape is (n_trials, 1500), add channel axis:
        if X_fix_list:
            # stack into (n_fix_trials, 1, window_len)
            X_val_fix = np.stack(X_fix_list)[:, None, :]
            y_val_fix = np.array(y_fix_list)
        else:
            # no clean fixation trials → empty arrays
            X_val_fix = np.empty((0, 1, window_len))
            y_val_fix = np.empty((0,), dtype=int)
        
        
        # require enough trials & ≥2 classes
        if (len(y_train) < 5 or len(set(y_train)) < 2 or
            len(y_val_dp) < 5 or len(y_val_enc) < 5 or len(y_val_fix) < 5):
            print(
                f"Skipping subject {subject}, channel {ch}: "
                f"train={len(y_train)}, "
                f"encoding={len(y_val_enc)}, "
                f"delay={len(y_val_dp)}, "
                f"fixation={len(y_val_fix)}"
            )
            continue
        
        # One‐vs‐all binary training per class
        for L in np.unique(y_train):
            # Get indices for positive (class L) and negative (all others)
            pos_idx = np.where(y_train == L)[0]
            neg_idx = np.where(y_train != L)[0]
            
            # require enough positives & negatives
            if len(pos_idx) < 5 or len(neg_idx) < 5:
                continue
            
            # Downsample negatives (stratified understampling) to match #positives, with equal proportion of each negative trial type (since we turned 4 classes into a one vs all analysis)
            # Try stratified undersampling; if it fails, do a simple random draw
            try:
                neg_sample = train_test_split(
                    neg_idx,
                    train_size=len(pos_idx),
                    stratify=y_train[neg_idx],
                    random_state=42
                )[0]
            except ValueError:
                # fallback: random undersample of all non-L trials
                neg_sample = np.random.choice(
                    neg_idx,
                    size=len(pos_idx),
                    replace=False
                )
            
            keep = np.hstack([pos_idx, neg_sample])
            X_bin = X_train[keep]                         # shape: (2*len(pos_idx),1,window_len)
            y_bin = (y_train[keep] == L).astype(int)      # 1 for L, 0 for others
            
            # Fit binary classifier
            pipe = make_pipeline(
                MiniRocket(num_kernels=10000, random_state=42),
                RidgeClassifierCV(alphas=np.logspace(-3,3,10))
            )
            pipe.fit(X_bin, y_bin)
            
            # Evaluate on each epoch: fixation, encoding, delay
            y_fix_bin = (y_val_fix == L).astype(int)
            y_enc_bin = (y_val_enc == L).astype(int)
            y_dp_bin  = (y_val_dp  == L).astype(int)
            
            acc_train = accuracy_score(y_bin,      pipe.predict(X_bin))
            acc_fix   = accuracy_score(y_fix_bin, pipe.predict(X_val_fix))
            acc_enc   = accuracy_score(y_enc_bin, pipe.predict(X_val_enc))
            acc_dp    = accuracy_score(y_dp_bin,  pipe.predict(X_val_dp))
            
            # Store one row per binary‐model in results
            results.append({
                'Subject':           subject,
                'ChannelIndex':      ch,
                'ROI':               region_labels[ch],
                'ClassLabel':        L,
                'TrainAccuracy':     round(acc_train, 3),
                'FixationAccuracy':  round(acc_fix,   3),
                'EncodingAccuracy':  round(acc_enc,   3),
                'DelayAccuracy':     round(acc_dp,    3),
            })

elapsed = time() - t0
print(f"Done in {elapsed:.1f}s — {len(results)} channel-ROIs processed.")

# 4) Wrap up


results_df = pd.DataFrame(results)
results_df = results_df.round(3)

# Keep only rows with all 4 classes present (i.e. length 4 TPR lists)
# results_df['TPR_per_class_Delay'] = results_df['TPR_per_class_Delay'].apply(lambda lst: [round(v, 3) for v in lst])
# results_df['TNR_per_class_Delay'] = results_df['TNR_per_class_Delay'].apply(lambda lst: [round(v, 3) for v in lst])
# mask = results_df['TPR_per_class_Delay'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 4)
# results_df = results_df[mask].reset_index(drop=True)
# print(f"Kept {len(results_df)} channels with all 4 conditions")

print(results_df)
print(results_df.sort_values(by='DelayAccuracy', ascending=False).head(10))

#  save
os.makedirs(os.path.join(model_results_dir, '09_onevsall'), exist_ok=True)

results_df.to_csv(os.path.join(model_results_dir, '09_onevsall/ENCTRAIN_noholdout_onevsall.csv'),
                  index=False)
