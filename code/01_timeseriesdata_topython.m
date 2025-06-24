% Extract raw preprocessed time series data for python import for Minirockets 10k
% convolutions time series classification

close all
clear
restoredefaultpath
addpath('/cluster/software/src/matlab_fieldtrip/fieldtrip-20240515')
addpath('/cluster/VAST/bkybg-lab/Data/k12wm/Coding/00apis/NPMK/')
addpath('/cluster/VAST/bkybg-lab/Data/k12wm/Coding/k12wm_preprocessing/functions/')
addpath('/cluster/VAST/bkybg-lab/Data/k12wm/Coding/k12wm_analysis/functions/')
cd('/cluster/VAST/bkybg-lab/Data/k12wm/')
ft_defaults
rawroot = '/cluster/VAST/bkybg-lab/Data/k12wm_rawdata';
root = '/cluster/VAST/bkybg-lab/Data/k12wm';

% Read in a list of subjects to iterate thru
subjects = readcell(['/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/code/k12wm_loop_turtles.csv'], 'Delimiter', ',', 'NumHeaderLines', 1);
version = 'uV_allregions'

% Create output directory
outdir = '/cluster/VAST/bkybg-lab/Data/project_repos/phzhr_turtles_av_ml/data/';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end


% Start from row
for subjIdx = 1:height(subjects)
    subject = subjects{subjIdx, 1};
    session = subjects{subjIdx, 2};
    
    fprintf('Processing: %s %s (%d/%d)\n', subject, session, subjIdx, height(subjects));

    subjpath = fullfile(root, subject, [subject '_' session]);

    try
        load(fullfile(subjpath, [subject '_' session '_1kft_notch_epoch_outliers_bip.mat']), 'ftBipolar');
        load(fullfile(subjpath, [subject '_' session '_labelsAnat.mat']), 'bipolarAnat');
        load(fullfile(subjpath, [subject '_' session 'gammamodchans.mat']), 'sigchans1', 'sigchans1_labels', 'sigchans2', 'sigchans2_labels', 'taskKey');
    catch ME
        warning('L Could not load files for %s %s: %s', subject, session, ME.message);
        continue
    end

    data = ftBipolar.trial;
    trialinfo = ftBipolar.trialinfo;
    nTrials = numel(data);
    nChannels = size(bipolarAnat, 1);
    nTime = size(data{1}, 2);

    % Preallocate array: trials × channels × time
    data_mat = zeros(nTrials, nChannels, nTime);
    for tr = 1:nTrials
        data_mat(tr, :, :) = data{tr}(1:nChannels, :);
    end

    % Grab anatomical labels (col 2 and 5 of bipolarAnat)
    anatomicallabels = bipolarAnat(:, [1, 5]);
    if istable(anatomicallabels)
        anatomicallabels = table2cell(anatomicallabels);
    end
    % Save to file
    outfile = fullfile(outdir, sprintf('%s_%s_%s.mat', subject, session, version));
    
    if isa(sigchans1_labels,'string')
      sigchans1_labels = cellstr(sigchans1_labels);
    end
    if isa(sigchans2_labels,'string')
      sigchans2_labels = cellstr(sigchans2_labels);
    end

% 3) Save as legacy v7 MAT-file (so scipy.loadmat can read it)
    save(outfile, 'data_mat', 'trialinfo', 'anatomicallabels', ...
        'sigchans1','sigchans1_labels', ...
        'sigchans2','sigchans2_labels', ...
        'taskKey', '-v7');

    clear ftBipolar bipolarAnat data_mat trialinfo anatomicallabels
end

fprintf('Done! Files written to: %s\n', outdir);