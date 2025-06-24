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
ROIlistfile = fullfile('E:\data', 'project_repos', 'phzhr_turtles_av_ml', 'model_results', 'region_subject_counts.csv');
subjects        = readcell(subjectListFile, 'Delimiter', ',', 'NumHeaderLines', 1);
ROIlist = readcell(ROIlistfile, 'Delimiter', ',', 'NumHeaderLines', 1);
ClassLookup = {'Color', 'Orientation', 'Tone', 'Duration'};
% ROI = {'L Inferior Temporal Gyrus'};

for i = 1:height(ROIlist)
    ROI = {ROIlist{i,3}};
    for j = 1:height(subjects)
        subject = subjects{j, 1};
        session = subjects{j, 2};
        load([root '/' subject '/' subject '_' session '/' subject '_' session '_1kft_notch_epochiti_outliers_bip_demean.mat']);
    
        % This loads the 'thresholds' table: [chan, trial, ..., isNaN]
    
        %% Define channels and trials of interest
        
        chanIdx = determineChannels_v3(subject, session, root, ROI);
        if isempty(chanIdx)
            disp(['Skipping ' subject ' ' session ' (no channels in ROI)']);
            continue;
        end
        trialLabels = ftDemean.trialinfo(:, 7);          % channel indices
        for L = 1:4
            freqPerChan = {};  % store TFRs for ROI channels in this class
            chanLabels = {};
        
            for k = chanIdx
                trialsToUse = find(trialLabels == L);
                validTrials = [];
        
                for t = trialsToUse'
                    trialData = ftDemean.trial{t};  % [nChannels x nTimepoints]
                    if any(isnan(trialData(k, :)))
                        continue;
                    end
                    validTrials(end+1) = t;
                end
        
                if isempty(validTrials)
                    disp(['Skipping channel ' ftDemean.label{k} ' (no valid trials)']);
                    continue;
                end
        
                disp(['Using ' num2str(length(validTrials)) ' valid trials for channel ' ftDemean.label{k}]);
        
                %% Select data
                cfg = [];
                cfg.channel = ftDemean.label(k);
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
                cfg.foi        = 4:1:40;
                % 5 cycles per frequency, so variable time window
                cfg.t_ftimwin  = 5 ./ cfg.foi;
                % Scale tapers to frequency as well
                cfg.tapsmofrq  = 0.3 * cfg.foi;
                cfg.toi        = 0:0.05:8;
                cfg.keeptrials = 'no';
                cfg.pad        = 'maxperlen';
                cfg.channel    = 'all';
        
                freq = ft_freqanalysis(cfg, dataSel);
        
                %% Baseline correction
                cfg = [];
                cfg.baseline     = [1.0 1.5];
                cfg.baselinetype = 'zscore';
                cfg.parameter    = 'powspctrm';
                freq = ft_freqbaseline(cfg, freq);
        
                % Store for ROI average later
                freqPerChan{end+1} = freq; 
                chanLabels{end+1}  = ftDemean.label{k};  
            end
        
            %% Average across ROI channels (if any)
            if ~isempty(freqPerChan)
                % Step 1: Extract the powspctrm fields into a cell array
                powCells = cellfun(@(x) x.powspctrm, freqPerChan, 'UniformOutput', false);  % 1xN cell of [1 x F x T]
                
                % Step 2: Concatenate over the 1st dim (channels)
                powArray = cat(1, powCells{:});  % [N x F x T]
                
                % Step 3: Store per class (1 figure for 4 subplots)
                avgPow = squeeze(mean(powArray, 1, 'omitnan'));  % [F x T]
            
                % Create subplot figure
                if L == 1
                    figure;
                    tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');
                    clim = [-3 3];  % consistent color range
                end
            
                nexttile;
                imagesc(freq.time, freq.freq, avgPow);
                axis xy;
                colormap jet;
                caxis(clim);
                title(ClassLookup{L});
                xlabel('Time (s)');
                ylabel('Frequency (Hz)');
                xline(2, '--k', 'Encoding', 'LabelOrientation', 'horizontal', ...
                      'LabelVerticalAlignment', 'bottom', 'FontSize', 10);
                xline(3.5, '--k', 'Delay Period', 'LabelOrientation', 'horizontal', ...
                      'LabelVerticalAlignment', 'bottom', 'FontSize', 10);
            
                if L == 4  % Add shared title + colorbar on last subplot
                    h = colorbar('Position', [0.93 0.11 0.02 0.815]);  % adjust as needed
                    ylabel(h, 'Z-scored Power (relative to 1–1.5s)', 'FontSize', 11);
                    sgtitle({[ROI{1} ' ' subject ' ' session], ...
                             [num2str(length(chanIdx)) ' chans']});
                end
            end
        end
    end
    saveplots_v_ppl(['E:\data\project_repos\phzhr_turtles_av_ml\model_results\19_ERSPs_PerSubj\' ROI{1}]);
    close all;
end