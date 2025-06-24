import os
import scipy.io
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import binomtest
import re

# IMPORT ALL THE DATA FROM THE MAT CELL ARRAYS
data_dir = '/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/data'
model_results_dir = '/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/model_results'
file_suffix = '_uV_allregions.mat'

# Collect matching .mat files
mat_files = [f for f in os.listdir(data_dir) if f.endswith(file_suffix)]

# Dictionary to store: region  set of subject IDs
region_subjects = defaultdict(set)

for fname in mat_files:
    fpath = os.path.join(data_dir, fname)
    try:
        mat = scipy.io.loadmat(fpath)
        labels = mat['anatomicallabels']  # shape: (n_channels, 2), dtype=object
        subject_id = fname.split('_')[0]  # Extract subject ID, e.g., "k12wm002"

        # Get all unique region labels (column 1 = index 1)
        regions_in_subject = set(str(row[1]).strip() for row in labels if str(row[1]).strip())

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


## FILTER REGIONS TO THOSE THAT OVERLAP IN 3 PEOPLE OR MORE
df_filtered = df[(df['SubjectCount'] >= 3) & (df['Region'] != 'Unknown')]

# Define the set of regions to include
included_regions = set(df_filtered['Region'])

# Clean up formatting
# Example raw included_regions:
# {"['L Middle Temporal Gyrus']", "['L IFG (p Opercularis)']", "['Unknown']", '[]', ...}

clean_regions = set()
for raw in included_regions:
    # remove any [, ], or ' characters
    name = re.sub(r"[\[\]']", "", raw).strip()
    if name and name != "Unknown":
        clean_regions.add(name)

included_regions = clean_regions

print(f"Kept {len(included_regions)} ROIs:\n", sorted(included_regions))

# Get list of all subjects that have any of these regions
top_group = sorted({subj for subjects in df_filtered['Subjects'] for subj in subjects})

# Now you can print or inspect:
print(f"Including {len(included_regions)} regions present in e3 subjects (excluding 'Unknown'):")
print(included_regions)
print(f"Iterating over {len(top_group)} subjects: {top_group}")

# === PREPARE MiniRocket Input ===

# Final storage for flattened rows
X_flat = []  # shape: (n_total_rows, 3000)
y_flat = []  # corresponding labels
trial_ids = []  # optional: store (subject, trial index) for tracking

for subject in top_group:
    subj_file = next((f for f in mat_files if f.startswith(subject)), None)
    if not subj_file:
        print(f"No file found for subject {subject}")
        continue

    fpath = os.path.join(data_dir, subj_file)
    mat = scipy.io.loadmat(fpath)

    data = mat['data_mat']
    trialinfo = mat['trialinfo']
    labels = mat['anatomicallabels']

    region_labels = [str(row[1]).strip().strip("[]'") for row in labels]
    selected_channel_indices = [i for i, label in enumerate(region_labels) if label in included_regions]

    if not selected_channel_indices:
        print(f"No matching ROIs found in {subject}")
        continue

    for i in range(data.shape[0]):
        if np.isnan(trialinfo[i, 0]) or np.isnan(trialinfo[i, 6]):
            continue

        dp_start = int(trialinfo[i, 3])
        trial_type = int(trialinfo[i, 6])

        start_idx = dp_start + 100
        end_idx = start_idx + 3000
        if end_idx > data.shape[2]:
            continue

        trial_slice = data[i, selected_channel_indices, start_idx:end_idx]

        # Keep only valid channels (no NaNs)
        for ch_idx in range(trial_slice.shape[0]):
            ch_data = trial_slice[ch_idx, :]
            if not np.isnan(ch_data).any():
                X_flat.append(ch_data)
                y_flat.append(trial_type)
                trial_ids.append((subject, i))  # optional: for provenance

X_flat = np.stack(X_flat, axis=0)  # shape: (n_trials × channels, 3000)
y_flat = np.array(y_flat)


# ## RUN MINIROCKET DUMB AND BROAD: Minirocket on a flat array of all subjects and ROI channels

# # Reshape for sktime: (n_instances, 1, timepoints)
# X_sktime = X_flat[:, np.newaxis, :]  # if you flattened by trial x channel

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_sktime, y_flat, test_size=0.2, random_state=42, stratify=y_flat)

# # Build pipeline
# pipeline = make_pipeline(
#     MiniRocket(num_kernels=10000),
#     RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# )

# # Fit model
# pipeline.fit(X_train, y_train)

# # Predict
# y_pred = pipeline.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc:.3f}")

# print("Classification Report:")
# print(classification_report(y_test, y_pred, digits=3))

### RUN MINIROCKET SMART: PER-SUBJECT, PER-CHANNEL MiniRocket with 70/20/10 split ===

results = []

print("Starting MiniRocket loop...\n")
start_time = time()

for subject in top_group:
    subj_file = next((f for f in mat_files if f.startswith(subject)), None)
    if not subj_file:
        print(f"Skipping subject {subject}: file not found.")
        continue
    print(f"Loading data for subject {subject}...")
    mat = scipy.io.loadmat(os.path.join(data_dir, subj_file))
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
        print(f"Skipping subject {subject}: not enough valid trials.")
        continue

    print(f"Splitting trials for subject {subject}...")
    train_trials, temp_trials = train_test_split(
        valid_trials, test_size=0.3, random_state=42, stratify=[int(trialinfo[t,6]) for t in valid_trials]
    )
    val_trials, test_trials = train_test_split(
        temp_trials, test_size=1/3, random_state=42,
        stratify=[int(trialinfo[t,6]) for t in temp_trials]
    )

    for ch in roi_chs:
        roi_name = region_labels[ch]
        print(f"  Subject {subject} | Channel {ch} | ROI {roi_name}: extracting trial data...")

        Xc, yc, tkeys = [], [], []
        for t in valid_trials:
            dp0 = int(trialinfo[t,3]) + 100
            ty = int(trialinfo[t,6])
            tslice = data[t, ch, dp0:dp0+3000]
            if not np.isnan(tslice).any():
                Xc.append(tslice)
                yc.append(ty)
                tkeys.append(t)
        if len(Xc) < 20 or len(set(yc)) < 2:
            print(f"  Skipping channel {ch}: not enough data or class imbalance.")
            continue

        Xc = np.stack(Xc)
        yc = np.array(yc)
        train_mask = np.isin(tkeys, train_trials)
        val_mask = np.isin(tkeys, val_trials)
        X_train = Xc[train_mask][:, None, :]
        y_train = yc[train_mask]
        X_val = Xc[val_mask][:, None, :]
        y_val = yc[val_mask]
        if len(y_train) < 5 or len(y_val) < 5:
            print(f"  Skipping channel {ch}: not enough samples in train/val.")
            continue

        print(f"  Training MiniRocket model for subject {subject}, channel {ch}, ROI {roi_name}...")
        pipe = make_pipeline(
            MiniRocket(num_kernels=10000, random_state=42),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        )
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        cm = confusion_matrix(y_val, y_val_pred, labels=sorted(np.unique(y_val)))
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - (tp + fn + fp)
        tpr_list = (tp / (tp + fn + 1e-6)).tolist()
        tnr_list = (tn / (tn + fp + 1e-6)).tolist()

        results.append({
            'Subject': subject,
            'ChannelIndex': ch,
            'ROI': roi_name,
            'TrainAccuracy': train_acc,
            'ValidationAccuracy': val_acc,
            'ConfusionMatrix': cm.tolist(),  
            'TPR_per_class': tpr_list,
            'TNR_per_class': tnr_list
        })

end_time = time()
elapsed = end_time - start_time
print(f"\nMiniRocket loop completed in {elapsed:.2f} seconds.")

results_df = pd.DataFrame(results)
results_df = results_df.round(3)
results_df['TPR_per_class'] = results_df['TPR_per_class'].apply(lambda lst: [round(v, 3) for v in lst])
results_df['TNR_per_class'] = results_df['TNR_per_class'].apply(lambda lst: [round(v, 3) for v in lst])
print(results_df)
print(results_df.sort_values(by='ValidationAccuracy', ascending=False).head(10))

# Keep only rows with all 4 classes present (i.e. length4 TPR lists)
mask = results_df['TPR_per_class'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 4)
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
# Assumes you have a DataFrame named results_df with column 'TPR_per_class'
# where each row is a list of TPRs for classes [1,2,3,4]
si_values = results_df['TPR_per_class'].apply(compute_si)

# Expand SI values into separate columns
for idx in range(4):
    results_df[f'SI_class{idx+1}'] = si_values.apply(lambda x: x[idx])

## TEST TOP CHANNELS ON TEST DATASET AFTER RETRAINING ON VALID AND TEST

# Filter the top performing channels 
filtered_df = results_df[
    (results_df['TrainAccuracy'] > 0.6) & (results_df['ValidationAccuracy'] > 0.4)
].copy()

test_accuracies = []
test_conf_matrices = []
test_tprs = []
test_tnrs = []

print("Starting MiniRocket Test Evaluation Loop...\n")
test_start = time()

for idx, row in filtered_df.iterrows():
    subject = row['Subject']
    ch = row['ChannelIndex']
    roi = row['ROI']
    
    print(f"Subject: {subject} | Channel: {ch} | ROI: {roi} ...", end=' ')
    
    try:
        subj_file = next((f for f in mat_files if f.startswith(subject)), None)
        mat = scipy.io.loadmat(os.path.join(data_dir, subj_file))
        data = mat['data_mat']
        trialinfo = mat['trialinfo']

        valid_trials = [
            t for t in range(data.shape[0])
            if not (np.isnan(trialinfo[t, 0]) or np.isnan(trialinfo[t, 6]))
            and int(trialinfo[t, 3]) + 3100 <= data.shape[2]
        ]
        train_trials, temp_trials = train_test_split(
            valid_trials, test_size=0.3, random_state=42,
            stratify=[int(trialinfo[t, 6]) for t in valid_trials]
        )
        val_trials, test_trials = train_test_split(
            temp_trials, test_size=1/3, random_state=42,
            stratify=[int(trialinfo[t, 6]) for t in temp_trials]
        )

        Xc, yc, tkeys = [], [], []
        for t in valid_trials:
            dp0 = int(trialinfo[t, 3])+100
            ty = int(trialinfo[t, 6])
            tslice = data[t, ch, dp0:dp0 + 3000]
            if not np.isnan(tslice).any():
                Xc.append(tslice)
                yc.append(ty)
                tkeys.append(t)

        Xc = np.stack(Xc)
        yc = np.array(yc)
        test_mask = np.isin(tkeys, test_trials)
        trainval_mask = np.isin(tkeys, train_trials + val_trials)
        X_test = Xc[test_mask][:, None, :]
        y_test = yc[test_mask]
        X_trainval = Xc[trainval_mask][:, None, :]
        y_trainval = yc[trainval_mask]

        pipe = make_pipeline(
            MiniRocket(num_kernels=10000, random_state=42),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        )
        pipe.fit(X_trainval, y_trainval)
        y_test_pred = pipe.predict(X_test)

        test_acc = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred,
                      labels=[1, 2, 3, 4])
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - (tp + fn + fp)
        tpr_list = (tp / (tp + fn + 1e-6)).tolist()
        tnr_list = (tn / (tn + fp + 1e-6)).tolist()

        test_accuracies.append(round(test_acc, 3))
        test_conf_matrices.append(cm.tolist())
        test_tprs.append([round(v, 3) for v in tpr_list])
        test_tnrs.append([round(v, 3) for v in tnr_list])
        
        print(f"Test Accuracy: {round(test_acc, 3)}")
    
    except Exception as e:
        print(f"L Failed: {e}")
        test_accuracies.append(np.nan)
        test_conf_matrices.append(None)
        test_tprs.append(None)
        test_tnrs.append(None)

# Add columns to DataFrame
filtered_df['TestAccuracy'] = test_accuracies
filtered_df['TestConfusionMatrix'] = test_conf_matrices
filtered_df['TestTPR_per_class'] = test_tprs
filtered_df['TestTNR_per_class'] = test_tnrs

#Move test accuracy to be next to validation accuracy for easy comparison by eye
cols = list(filtered_df.columns)
cols.remove('TestAccuracy')
insert_idx = cols.index('ValidationAccuracy') + 1
cols.insert(insert_idx, 'TestAccuracy')
filtered_df = filtered_df[cols]

elapsed_test = time() - test_start
print(f"\nTest evaluation loop completed in {elapsed_test:.2f} seconds.")
print(filtered_df)



## Look at Selectivity in validation vs test accuracy for each condition w/ significance.  Make a shorter DF of sig channels class selective

# Initialize new columns
for i in range(1,5):
    filtered_df[f'p_val_class{i}'] = np.nan
    filtered_df[f'selective_and_sig_class{i}'] = False

# Perform binomial tests per class
for idx, row in filtered_df.iterrows():
    cm = np.array(row['TestConfusionMatrix'])
    for i in range(4):
        tp = cm[i, i]
        n_i = cm[i, :].sum()
        # Binomial test against chance p=0.25
        if n_i > 0:
            pval = binomtest(tp, n_i, p=0.25, alternative='greater').pvalue
        else:
            pval = np.nan
        filtered_df.at[idx, f'p_val_class{i+1}'] = round(pval, 4)
        # check selectivity and significance
        si = row[f'SI_class{i+1}']
        filtered_df.at[idx, f'selective_and_sig_class{i+1}'] = (si > 0) and (pval < 0.2)


# Show results for channels that are selective and significant for any class
sig_mask = filtered_df[[f'selective_and_sig_class{i}' for i in range(1,5)]].any(axis=1)
sig_df = filtered_df[sig_mask]

print(sig_df)


#  sav
results_df.to_csv(os.path.join(model_results_dir, 'DELAYPERIOD_minirocket_channel_val_accuracies_model4.csv'),
                  index=False)

filtered_df.to_csv(os.path.join(model_results_dir, 'DELAYPERIOD_mminirocket_channel_subset_test_accuracies_model4.csv'),
                  index=False)

sig_df.to_csv(os.path.join(model_results_dir, 'DELAYPERIOD_mminirocket_channel_selective_channels_model4.csv'),
                  index=False)
