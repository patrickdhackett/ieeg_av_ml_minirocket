import os
import scipy.io
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.model_selection import train_test_split

data_dir = '/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/data'
file_suffix = '_uV_allregions.mat'

# Collect matching .mat files
mat_files = [f for f in os.listdir(data_dir) if f.endswith(file_suffix)]

# Dictionary to store: region  set of subject IDs
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
print(df)


## Filter regions with at least 5 subjects
df_filtered = df[df['SubjectCount'] >= 5]

# Get list of all unique subjects in these regions
all_subjects = sorted({s for subject_list in df_filtered['Subjects'] for s in subject_list})

# Count overlap of all 5-subject combinations across regions
group_counts = Counter()
for combo in combinations(all_subjects, 5):
    count = sum(set(combo).issubset(set(subjects)) for subjects in df_filtered['Subjects'])
    if count > 0:
        group_counts[combo] = count

# Find the top 5-subject group with most region overlap
top_group, top_overlap = group_counts.most_common(1)[0]

# Get regions that all 5 subjects share
shared_regions = df_filtered[df_filtered['Subjects'].apply(lambda subs: set(top_group).issubset(set(subs)))]['Region'].tolist()

# Display results
print("Top 5 subjects with most overlapping regions:")
print("Subjects:", top_group)
print("Number of shared regions:", top_overlap)
print("Shared Regions:")
for region in shared_regions:
    print(" -", region)


# === PREPARE MiniRocket Input ===

included_regions = {
    'L Middle Temporal Gyrus',
    'L Fusiform Gyrus',
    'L ParaHippocampal Gyrus',
    'L Inferior Temporal Gyrus',
    'L IFG (p Triangularis)',
    'L Superior Temporal Gyrus'
}


# Final storage for flattened rows
X_flat = []  # shape: (n_total_rows, 5500)
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

        enc_start = int(trialinfo[i, 0])
        trial_type = int(trialinfo[i, 6])

        start_idx = enc_start
        end_idx = start_idx + 5500
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

X_flat = np.stack(X_flat, axis=0)  # shape: (n_trials × channels, 5500)
y_flat = np.array(y_flat)


# Run Minirocket

# Reshape for sktime: (n_instances, 1, timepoints)
X_sktime = X_flat[:, np.newaxis, :]  # if you flattened by trial x channel

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_sktime, y_flat, test_size=0.2, random_state=42, stratify=y_flat)

# Build pipeline
pipeline = make_pipeline(
    MiniRocket(num_kernels=10000),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)

# Fit model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))