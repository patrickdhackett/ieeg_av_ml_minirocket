%% turtles_gamma_analysis.m
% Load turtles data, compute high-gamma power per condition, and plot

close all
clear
restoredefaultpath

% add FieldTrip and project paths
addpath('C:/Users/phzhr/OneDrive - University of Missouri/Files/Coding/00apis/fieldtrip-20240309');
addpath('C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/k12wm/Coding/00apis/NPMK');
addpath('C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/k12wm/Coding/k12wm_preprocessing/functions');
addpath('C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/k12wm/Coding/k12wm_analysis/functions');
ft_defaults

root            = 'C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/k12wm';
subjectListFile = fullfile('C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/', 'project_repos', 'phzhr_turtles_av_ml', 'code', 'k12wm_loop_turtles.csv');
subjects        = readcell(subjectListFile, 'Delimiter', ',', 'NumHeaderLines', 1);

% which column in trialinfo holds the condition ID
condIdxCol = 7;


% Loop thru the subjects and load their data and do the necessary
% transformations and plotting.
for iSub = 11:11%(subjects,1)
    subj = subjects{iSub,1};
    sess = subjects{iSub,2};
    fprintf('Processing %s_%s (%d/%d)\n', subj, sess, iSub, size(subjects,1));

    subjPath = fullfile(root, subj, [subj '_' sess]);
    try
        D = load(fullfile(subjPath, [subj '_' sess '_1kft_notch_epoch_outliers_bip.mat']), 'ftBipolar');
        L = load(fullfile(subjPath, [subj '_' sess '_labelsAnat.mat']),       'bipolarAnat');
        M = load(fullfile(subjPath, [subj '_' sess 'gammamodchans.mat']),     ...
                  'sigchans1', 'sigchans2', 'taskKey');
    catch ME
        warning('  skipped %s_%s: %s', subj, sess, ME.message);
        continue;
    end

    ftBipolar   = D.ftBipolar;
    bipolarAnat = L.bipolarAnat;
    condNames   = M.taskKey;    % e.g. {'Color';'Orientation';'Tone';'NoiseDuration'}

    nChans  = size(bipolarAnat,1);
    nTrials = numel(ftBipolar.trial);
    nTime   = size(ftBipolar.trial{1},2);

    % get condition ID per trial
    trialConds = ftBipolar.trialinfo(:,condIdxCol);
    nConds     = numel(M.sigchans1);


    chans = cell(4,1);
    for c = 1:4
        chans{c} = intersect(M.sigchans1{c}, M.sigchans2{c});
    end
    
    % 2) get the unique channels across *all* conditions
    allChans = unique([chans{:}]);
    
    % TFR parameters
    cfgTFR              = [];
    cfgTFR.method       = 'wavelet'; % can do'wavelet' or 'mtmconvol' multitapers with smoothing but requires more tuning, results look teh same
    cfgTFR.output       = 'pow';
    cfgTFR.foi          = 1:2:150;             % 1–150 Hz in 2-Hz steps
    cfgTFR.toi = ftBipolar.time{1}(1:10:end);       % For the whole time series in 10 ms steps (ftBipolar.time is 0 to 6.999 secs )
    cfgTFR.pad =     'nextpow2';
    if strcmp(cfgTFR.method,'wavelet')
        cfgTFR.width        = 7;              % 7‐cycle Morlet wavelets across all freqs

    elseif strcmp(cfgTFR.method,'mtmconvol')
        cfgTFR.t_ftimwin    = 5 ./ cfgTFR.foi;     % 5 cycles per frequency
        cfgTFR.tapsmofrq    = 0.2 * cfgTFR.foi;    % ±20% smoothing
    end
    % baseline‐correction params
    cfgBL                      = [];
    cfgBL.baseline             = [0.1 0.9];    % grab the middle of fixation
    cfgBL.baselinetype         = 'relchange';    % percent change
    % 3) for each channel, find which conditions it belongs to
    for u = 1:numel(allChans)
      chanID   = allChans(u);
      chanName = bipolarAnat.ROI{chanID};
      anatLbl = char(bipolarAnat.anatmacro1(chanID));
      condIdxs   = find(cellfun(@(c) ismember(chanID,c), chans));
      condLabels = condNames(condIdxs,2)';
      condStr    = strjoin(condLabels,', ');
      % find which trials are non-NaN for this channel
      isGood = cellfun(@(x) ~any(isnan(x(chanID,:))), ftBipolar.trial).';
        
      figure('Name',sprintf('ERSP_%s_chan%d_%s',subj,chanID,anatLbl), 'NumberTitle','off','Color','w');
            sgtitle({ ...
              sprintf('%s — %s_chan%s', subj, chanName, num2str(chanID)), ...
              sprintf('%s', anatLbl) ...
              sprintf('Γ-modulated conds: %s', strjoin(condLabels, ', ')) ...
            }, 'Interpreter','none');
        for cat = 1:4
            % only pick trials of this type *and* that have no NaNs
            selTrials = find(trialConds==cat & isGood);
            subplot(2,2, cat);
            if isempty(selTrials)
              warning('No valid (non-NaN) trials for channel %s, cond %s – skipping', ...
                      chanName, condNames{cat,2});
              continue;
            end
        
            % build cfg fresh so method/output/etc. are preserved
            cfg        = cfgTFR;             
            cfg.trials = selTrials;          
            cfg.channel= ftBipolar.label{chanID};
        
            % TFR + baseline
            tfr    = ft_freqanalysis(cfg, ftBipolar);
            tfr_bl = ft_freqbaseline(cfgBL, tfr);
        
            % now plot with a forced nonzero zlim to be safe
            dat = squeeze(tfr_bl.powspctrm(1,:,:));   % freq × time
    
            % 2) find the non-NaN, non-zero entries
            valid = dat(~isnan(dat));
            
            % 3) if there’s literally no data (all NaN), skip plotting
            if isempty(valid)
              warning('Channel %s, cond %s: no valid data → skipping plot', ...
                      chanName, condNames{cat,2});
              continue;
            end
            
            % 4) pick a color‐limit based on the extremes of the *valid* data
            m = max(abs(valid));
            if m == 0
              % all values are exactly zero → pick a small default or a fixed range
              m = 1;    % e.g. +/-1 unit (you can choose another fallback)
            end
            zlim = [-m m];
            
            % 5) now plot with a guaranteed valid zlim
             freqVec = tfr_bl.freq;
              timeVec = tfr_bl.time;
              imagesc(timeVec, freqVec, dat);
              axis xy;
              caxis(zlim);
              xlabel('Time (s)'); ylabel('Frequency (Hz)');
              nTrials = numel(selTrials);
              title(sprintf('ERSP: %s %d trials',condNames{cat,2}, nTrials), 'Interpreter','none');
              hold on;
              plot([1 1], ylim, 'k--');   % event marker at 1 s
              hold off;
              cb = colorbar;
              cb.Label.String = 'Spectral power (z-score)';
        end
    end
end
%saveplots_v_ppl_subplots('C:/Users/phzhr/OneDrive - University of Missouri/Shared Documents - Kundu Lab - Ogrp/data/project_repos/phzhr_turtles_av_ml/descriptive_analytics/04c_ERSP_k12wm010_gammamodchans')
fprintf('Processing complete.\n');
