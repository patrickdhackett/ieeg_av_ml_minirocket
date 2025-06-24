# This version 7 removes the train/val split.  We are training on encoding only, but for all trials, then we are validating (but not training model weights) for eval only on fix/enc/delay period

import os
import scipy.io
import pandas as pd
from collections import defaultdict
import numpy as np
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
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
        if (len(y_val_dp) < 5 or len(y_val_enc) < 5 or len(y_val_fix) < 5):
            print(
                f"Skipping subject {subject}, channel {ch}: "
                f"encoding={len(y_val_enc)}, "
                f"delay={len(y_val_dp)}, "
                f"fixation={len(y_val_fix)}"
            )
            continue
        
        # train & eval
        pipe = make_pipeline(
            MiniRocket(num_kernels=10000, random_state=42),
            RidgeClassifierCV(alphas=np.logspace(-3,3,10))
        )
        pipe.fit(X_val_dp, y_val_dp)
        # get three validation accuracies
        va_fix = accuracy_score(y_val_fix, pipe.predict(X_val_fix))
        va_enc = accuracy_score(y_val_enc, pipe.predict(X_val_enc))
        va_dp  = accuracy_score(y_val_dp,  pipe.predict(X_val_dp))
    
        # (optional) confusion matrix on delay
        cm_dp = confusion_matrix(
            y_val_dp,
            pipe.predict(X_val_dp),
            labels=sorted(np.unique(y_val_dp))
        )
        tp = np.diag(cm_dp)
        fn = cm_dp.sum(axis=1) - tp
        fp = cm_dp.sum(axis=0) - tp
        tn = cm_dp.sum() - (tp + fn + fp)
        tpr_dp = (tp/(tp+fn+1e-6)).round(3).tolist()
        tnr_dp = (tn/(tn+fp+1e-6)).round(3).tolist()
    
        results.append({
            'Subject': subject,
            'ChannelIndex': ch,
            'ROI': region_labels[ch],
            'FixationAccuracy':  round(va_fix,3),
            'EncodingAccuracy':  round(va_enc,3),
            'DelayAccuracy':     round(va_dp, 3),
            'ConfusionMatrixDelay': cm_dp.tolist(),
            'TPR_per_class_Delay':   tpr_dp,
            'TNR_per_class_Delay':   tnr_dp
        })

elapsed = time() - t0
print(f"Done in {elapsed:.1f}s — {len(results)} channel-ROIs processed.")

# 4) Wrap up


results_df = pd.DataFrame(results)
results_df = results_df.round(3)
results_df['TPR_per_class_Delay'] = results_df['TPR_per_class_Delay'].apply(lambda lst: [round(v, 3) for v in lst])
results_df['TNR_per_class_Delay'] = results_df['TNR_per_class_Delay'].apply(lambda lst: [round(v, 3) for v in lst])
print(results_df)
print(results_df.sort_values(by='DelayAccuracy', ascending=False).head(10))

# Keep only rows with all 4 classes present (i.e. length 4 TPR lists)
mask = results_df['TPR_per_class_Delay'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 4)
results_df = results_df[mask].reset_index(drop=True)

print(f"Kept {len(results_df)} channels with all 4 conditions")

# Add a selectivity measure for each class
# Define a function to compute Selectivity Index (SI) for a list of class-specific TPRs
def compute_si(tpr_list):
    tpr = np.array(tpr_list)
    C = len(tpr)
    sis = []
    for i in range(C):
        tpr_i = tpr[i]
        others = np.delete(tpr, i)
        mean_others = others.mean() if len(others) > 0 else 0
        denom = max(tpr_i, mean_others)
        si = (tpr_i - mean_others) / denom if denom > 0 else 0.0
        sis.append(round(si, 3))
    return sis

# Apply SI computation to your results_df
#-1 to 1 where 1 means it responds only to its own class, -1 it never responds to its own class
# Assumes you have a DataFrame named results_df with column 'TPR_per_class_Delay'
# where each row is a list of TPRs for classes [1,2,3,4]
si_values = results_df['TPR_per_class_Delay'].apply(compute_si)

# Expand SI values into separate columns
for idx in range(4):
    results_df[f'SI_class{idx+1}'] = si_values.apply(lambda x: x[idx])




#  save
os.makedirs(os.path.join(model_results_dir, '08_noholdout_evalallperiods'), exist_ok=True)

results_df.to_csv(os.path.join(model_results_dir, '08_noholdout_evalallperiods/DPTRAIN_evalfixencdp_minirocket_channel_val_accuracies.csv'),
                  index=False)
