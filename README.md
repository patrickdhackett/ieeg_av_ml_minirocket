**iEEG AV ML MiniRocket**

Time--frequency clustering & MiniRocket decoding of auditory-visual
working-memory iEEG

**Repository Structure**

ieeg_av_ml_minirocket/  
code/ Python scripts  
descriptive_analytics/ Exploratory analyses notebooks  
figures/ Static figure exports  
literature/ PDFs and summary tables  
model_results/ CSV outputs from analyses  
plots_ppl/ Phase-locking and PAC plots  
.gitignore Exclude data/, temp files, etc.  
README.doc (this file) Project overview

**Overview**

1.  Time--Frequency Clustering  
    -- Compute Morlet spectrograms  
    -- Baseline-correct to middle 1 000 ms ITI  
    -- Threshold and label clusters, export CSV

2.  Permutation-Shuffled MiniRocket + Ridge  
    -- One-vs-all classification per channel×class  
    -- Train on encoding (or on Delay1, adjustable)  
    -- Permutation p-values and binomial tests  
    -- Export detailed CSV of accuracies and significance

3.  Full-Trial Spectrograms  
    -- Average spectrogram per class in left IFG (pars opercularis) and
    STG  
    -- Dotted lines at 2000 ms (encoding start) and 3500 ms (delay
    start)

4.  Literature Summaries  
    -- Curated tables of IEEG working-memory studies (auditory, visual,
    object)  
    -- Machine-learning decoding papers

**Requirements**

• Python 3.8 or newer  
• Packages:  
numpy, scipy, pandas, matplotlib, mne, scikit-learn, sktime, tqdm,
joblib

Install via pip:

pip install numpy scipy pandas matplotlib mne scikit-learn sktime tqdm
joblib

**Data**

Not included in this repository. Add your MATLAB files under a local
data/ folder, named:  
\<subject\>*turtles*\<session\>\_uV_allregions_iti.mat

Each file must contain:  
• data_mat array (n_trials × n_channels × n_times)  
• trialinfo matrix with onset and class-label columns  
• anatomicallabels cell array of region names

**Usage**

1.  **Time--frequency clustering**  
    python code/14_timefreq_clusters.py

2.  **Permutation-shuffled decoding (encoding-trained)**  
    python code/13_permutationshuffledtrain_iti.py

3.  **Permutation-shuffled decoding (delay1-trained, test split)**  
    python code/15_permutationshuffledtrain_iti_testsplit.py

4.  **Spectrogram plotting**  
    python code/plot_spectrograms_fulltrial.py

Outputs (CSVs and PNGs) will be saved under model_results/ and figures/.

**Results**

All analysis outputs---cluster tables, classification results,
spectrograms---appear in:  
• model_results/  
• figures/
