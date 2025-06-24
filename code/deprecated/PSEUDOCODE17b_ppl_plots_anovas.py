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
DATA_DIR          = r'E:\data\project_repos\phzhr_turtles_av_ml\data\ppl'
MODEL_RESULTS_DIR = r'E:\data\project_repos\phzhr_turtles_av_ml\model_results'
FILE_SUFFIX       = '_ppl_iti.mat'
np.random.seed(42)               # reproducibility
thetaRange = [4, 8]
betaRange = [15, 25]
lowGamma = [26, 70]
highGamma = [70, 140]

# ──────────────────────────────────────────────────────────────────────────────

# Find all subject MAT files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(FILE_SUFFIX)]

# Load region counts across subjects (to only analyze regions occurring in 3 subj or more)
region_counts = pd.read_csv(DATA_DIR + '\\region_subject_counts.csv')

# Load the frequency map
freqs = pd.read_csv(DATA_DIR + '\\f1.csv')
# remove region counts occurring in less than 3 subjs
region_counts.drop(region_counts(region_counts[2]<3))

# Load per subj data
for fname in mat_files:
    mat  = scipy.io.loadmat(os.path.join(DATA_DIR, fname))
    PPL_dp_first1k = mat['PPL_dp_first1k']
    PPL_dp_second1k = mat['PPL_dp_second1k']
    PPL_dp_third1k = mat['PPL_dp_third1k']
    PPL_enc_first1k = mat['PPL_enc_first1k']
    PPL_fix_all = mat['PPL_fix_all']
    sigchans1 = mat['sigchans1']
    sigchans1_labels = mat['sigchans1_labels']
    sigchans2 = mat['sigchans2']
    sigchans2_labels = mat['sigchans2_labels']
    taskKey = mat['taskKey']
    trialinfo = mat['trialinfo']
    chancombo = mat['taskKey']
    combolabels = mat['trialinfo']       
    subj = fname.split('_')[0]
    
    # find all region pairs in region_counts remaining after region counts occurring in less than 3 subjs were removed in an earlier step
    for i in region_counts:
        for j in region_counts:
            regionA = i[1]
            regionB = j[2]
            # The 3rd column of combolabels contains the concatenated region names for the chancombo
            channel_idxs = find(combolabels(where regionA in combolabels(2) and regionB in combolabels(2))
                                
            # find idxs of each trial class
            L1_task_idxs = find(trialinfo(where 1 in trialinfo(6)))
            L2_task_idxs = find(trialinfo(where 2 in trialinfo(6)))
            L3_task_idxs = find(trialinfo(where 3 in trialinfo(6)))
            L4_task_idxs = find(trialinfo(where 4 in trialinfo(6)))
            
            # correct each time window per freq per trial against fixation
            PPL_dp1_corr = PPL_dp_first1k - PPL_fix_all
            PPL_dp2_corr = PPL_dp_first1k - PPL_fix_all
            PPL_dp3_corr = PPL_dp_first1k - PPL_fix_all
            PPL_enc_corr = PPL_dp_first1k - PPL_fix_all
            
            # Find indexs of each freq band of interest
            thetaidxs = find(freqs(freqs <= betaRange[2] OR freqs >= betaRange[1])
            betaidxs = find(freqs(freqs <= thetaRange[2] OR freqs >= thetaRange[1])
            lowgammaidxs = find(freqs(freqs <= lowGamma[2] OR freqs >= lowGamma[1])
            highgammaidxs = find(freqs(freqs <= highGamma[2] OR freqs >= highGamma[1])
            
            # plot theta band
            xaxis_labels = ['Encoding','Delay1','Delay2','Delay3']
            encmean_class1 = mean(PPL_enc_corr(L1_task_idxs, channel_idxs, thetaidxs))
            encstddev_class1 = stdev(PPL_enc_corr(L1_task_idxs, channel_idxs, thetaidxs))
            
            dp1_class1_mean = mean(PPL_dp1_corr(L1_task_idxs, channel_idxs, thetaidxs))
            dp1_class1_std = stdev(PPL_dp1_corr(L1_task_idxs, channel_idxs, thetaidxs))
            
            dp2_class1_mean = mean(PPL_dp2_corr(L1_task_idxs, channel_idxs, thetaidxs))
            dp2_class1_std = stdev(PPL_dp2_corr(L1_task_idxs, channel_idxs, thetaidxs))
            
            dp3_class1_mean = mean(PPL_dp3_corr(L1_task_idxs, channel_idxs, thetaidxs))
            dp3_class1_std = stdev(PPL_dp3_corr(L1_task_idxs, channel_idxs, thetaidxs))
            
            encmean_class2 = mean(PPL_enc_corr(L2_task_idxs, channel_idxs, thetaidxs))
            encstddev_class2 = stdev(PPL_enc_corr(L2_task_idxs, channel_idxs, thetaidxs))
            
            dp1_class2_mean = mean(PPL_dp1_corr(L2_task_idxs, channel_idxs, thetaidxs))
            dp1_class2_std = stdev(PPL_dp1_corr(L2_task_idxs, channel_idxs, thetaidxs))
            
            dp2_class2_mean = mean(PPL_dp2_corr(L2_task_idxs, channel_idxs, thetaidxs))
            dp2_class2_std = stdev(PPL_dp2_corr(L2_task_idxs, channel_idxs, thetaidxs))
            
            dp3_class2_mean = mean(PPL_dp3_corr(L2_task_idxs, channel_idxs, thetaidxs))
            dp3_class2_std = stdev(PPL_dp3_corr(L2_task_idxs, channel_idxs, thetaidxs))
            
            encmean_class3 = mean(PPL_enc_corr(L3_task_idxs, channel_idxs, thetaidxs))
            encstddev_class3 = stdev(PPL_enc_corr(L3_task_idxs, channel_idxs, thetaidxs))
            
            dp1_class3_mean = mean(PPL_dp1_corr(L3_task_idxs, channel_idxs, thetaidxs))
            dp1_class3_std = stdev(PPL_dp1_corr(L3_task_idxs, channel_idxs, thetaidxs))
            
            dp2_class3_mean = mean(PPL_dp2_corr(L3_task_idxs, channel_idxs, thetaidxs))
            dp2_class3_std = stdev(PPL_dp2_corr(L3_task_idxs, channel_idxs, thetaidxs))
            
            dp3_class3_mean = mean(PPL_dp3_corr(L3_task_idxs, channel_idxs, thetaidxs))
            dp3_class3_std = stdev(PPL_dp3_corr(L3_task_idxs, channel_idxs, thetaidxs))
            
            encmean_class4 = mean(PPL_enc_corr(L4_task_idxs, channel_idxs, thetaidxs))
            encstddev_class4 = stdev(PPL_enc_corr(L4_task_idxs, channel_idxs, thetaidxs))
            
            dp1_class4_mean = mean(PPL_dp1_corr(L4_task_idxs, channel_idxs, thetaidxs))
            dp1_class4_std = stdev(PPL_dp1_corr(L4_task_idxs, channel_idxs, thetaidxs))
            
            dp2_class4_mean = mean(PPL_dp2_corr(L4_task_idxs, channel_idxs, thetaidxs))
            dp2_class4_std = stdev(PPL_dp2_corr(L4_task_idxs, channel_idxs, thetaidxs))
            
            dp3_class4_mean = mean(PPL_dp3_corr(L4_task_idxs, channel_idxs, thetaidxs))
            dp3_class4_std = stdev(PPL_dp3_corr(L4_task_idxs, channel_idxs, thetaidxs))
            
            # x axis will have a bar for each class at each time window
            # need standard error whiskers on the bars
            # need dots 
            # need anova at each window across the classes
            # need to average across subjects
            # need a plot for each freq band each regionpair
            