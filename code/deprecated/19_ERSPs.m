close all
clear
restoredefaultpath

% Add FieldTrip and project paths
addpath('E:\data\k12wm\Coding\apis\fieldtrip-20250523');
addpath('E:\data\k12wm\Coding\k12wm_preprocessing\functions');
addpath('E:\data\k12wm\Coding\k12wm_analysis\functions');
ft_defaults

root            = 'E:\data\k12wm';
subjectListFile = fullfile('E:\data', 'project_repos', 'phzhr_turtles_av_ml', 'code', 'k12wm_loop_turtles.csv');
subjects        = readcell(subjectListFile, 'Delimiter', ',', 'NumHeaderLines', 1);
ClassLookup = {'Color', 'Orientation', 'Tone', 'Duration'};
for i = 1:3 %height(subjects)
    subject = subjects{i, 1};
    session = subjects{i, 2};
    load([root '/' subject '/' subject '_' session '/' subject '_' session '_1kft_notch_epochiti_outliers_bip_demean.mat']);

    % This loads the 'thresholds' table: [chan, trial, ..., isNaN]

    %% Define channels and trials of interest
    ROI = {'L Superior Temporal Gyrus'};
    chanIdx = determineChannels_v3(subject, session, root, ROI);
    if isempty(chanIdx)
        disp(['Skipping ' subject ' ' session ' (no channels in ROI)']);
        continue;
    end
    trialLabels = ftDemean.trialinfo(:, 7);          % channel indices
    for L = 1:4
        freqPerChan = {};  % store TFRs for ROI channels in this class
        chanLabels = {};
    
        for j = chanIdx
            trialsToUse = find(trialLabels == L);
            validTrials = [];
    
            for t = trialsToUse'
                trialData = ftDemean.trial{t};  % [nChannels x nTimepoints]
                if any(isnan(trialData(j, :)))
                    continue;
                end
                validTrials(end+1) = t;
            end
    
            if isempty(validTrials)
                disp(['Skipping channel ' ftDemean.label{j} ' (no valid trials)']);
                continue;
            end
    
            disp(['Using ' num2str(length(validTrials)) ' valid trials for channel ' ftDemean.label{j}]);
    
            %% Select data
            cfg = [];
            cfg.channel = ftDemean.label(j);
            cfg.trials  = validTrials;
            dataSel = ft_selectdata(cfg, ftDemean);
    
            %% Crop each trial to first 8 seconds
            for t = 1:length(dataSel.trial)
                tStart = dataSel.time{t}(1);
                tEnd   = tStart + 8;
                timeMask = dataSel.time{t} <= tEnd;
                dataSel.trial{t} = dataSel.trial{t}(:, timeMask);
                dataSel.time{t}  = dataSel.time{t}(timeMask);
            end
    
            %% Time-frequency analysis
            cfg = [];
            cfg.output     = 'pow';
            cfg.method     = 'mtmconvol';
            cfg.foi        = 2:2:100;
            cfg.t_ftimwin  = 5 ./ cfg.foi;
            cfg.tapsmofrq  = 0.4 * cfg.foi;
            cfg.toi        = 0:0.05:8;
            cfg.keeptrials = 'no';
            cfg.pad        = 'maxperlen';
            cfg.channel    = 'all';
    
            freq = ft_freqanalysis(cfg, dataSel);
    
            %% Baseline correction
            cfg = [];
            cfg.baseline     = [0.5 1.0];
            cfg.baselinetype = 'zscore';
            cfg.parameter    = 'powspctrm';
            freq = ft_freqbaseline(cfg, freq);
    
            % Store for ROI average later
            freqPerChan{end+1} = freq; 
            chanLabels{end+1}  = ftDemean.label{j};  
        end
    
        %% Average across ROI channels (if any)
        if ~isempty(freqPerChan)
            cfg = [];
            cfg.parameter = 'powspctrm';
            cfg.method    = 'average';
            freqAvg = ft_freqgrandaverage(cfg, freqPerChan{:});
    
            cfg = [];
            cfg.parameter = 'powspctrm';
            cfg.channel   = 'all';
            cfg.zlim      = 'maxabs';
            cfg.xlim      = [0 8];
            cfg.ylim      = [2 100];
    
            figure;
            ft_singleplotTFR(cfg, freqAvg);
            title([ROI{1} ' ' subject ' ' session ' Class:' ClassLookup(L)]);
        end
    end
end
