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
ClassLookup     = {'Color', 'Orientation', 'Tone', 'Duration'};
ROI             = {'L Inferior Temporal Gyrus'};

% Initialize container for each class
classPow = cell(1, 4);  % classPow{L} will be [N_chan_total x F x T]
metaFreq = [];  % store one freq struct for metadata reuse
allChans = {};
allSubjects = {};
for i = 1:height(subjects)
    subject = subjects{i, 1};
    session = subjects{i, 2};
    load(fullfile(root, subject, [subject '_' session], [subject '_' session '_1kft_notch_epochiti_outliers_bip_demean.mat']));

    chanIdx = determineChannels_v3(subject, session, root, ROI);
    if isempty(chanIdx)
        disp(['Skipping ' subject ' ' session ' (no channels in ROI)']);
        continue;
    end

    trialLabels = ftDemean.trialinfo(:, 7);

    for L = 1:4
        for j = chanIdx
            trialsToUse = find(trialLabels == L);
            validTrials = [];

            for t = trialsToUse'
                trialData = ftDemean.trial{t};
                if any(isnan(trialData(j, :)))
                    continue;
                end
                validTrials(end+1) = t;
            end

            if isempty(validTrials)
                continue;
            end
            allChans{end+1} = ftDemean.label{j};
            allSubjects{end+1} = subject;
            % Select data
            cfg = [];
            cfg.channel = ftDemean.label(j);
            cfg.trials  = validTrials;
            dataSel = ft_selectdata(cfg, ftDemean);

            % Crop each trial to first 8 seconds
            for t = 1:length(dataSel.trial)
                tStart = dataSel.time{t}(1);
                tEnd   = tStart + 8;
                timeMask = dataSel.time{t} <= tEnd;
                dataSel.trial{t} = dataSel.trial{t}(:, timeMask);
                dataSel.time{t}  = dataSel.time{t}(timeMask);
            end

            % Time-frequency analysis
            cfg = [];
            cfg.output     = 'pow';
            cfg.method     = 'mtmconvol';
            cfg.foi        = 4:1:40;
            cfg.t_ftimwin  = 5 ./ cfg.foi;
            cfg.tapsmofrq  = 0.3 * cfg.foi;
            cfg.toi        = 0:0.05:8;
            cfg.keeptrials = 'no';
            cfg.pad        = 'maxperlen';
            cfg.channel    = 'all';

            freq = ft_freqanalysis(cfg, dataSel);

            % Baseline correction
            cfg = [];
            cfg.baseline     = [1.0 1.5];
            cfg.baselinetype = 'zscore';
            cfg.parameter    = 'powspctrm';
            freq = ft_freqbaseline(cfg, freq);

            % Store powspctrm
            classPow{L}(end+1, :, :) = freq.powspctrm;

            % Save metadata struct from first example
            if isempty(metaFreq)
                metaFreq = freq;
            end
        end
    end
end

% Now average across subjects and channels for each class
for L = 1:4
    if isempty(classPow{L})
        continue;
    end

    avgPow = squeeze(mean(classPow{L}, 1, 'omitnan'));  % [F x T]

    % Reconstruct freqAvg
    freqAvg = metaFreq;
    freqAvg.powspctrm = reshape(avgPow, [1 size(avgPow)]);  % [1 x F x T]
    freqAvg.label = {['Group ROI avg: ' ROI{1}]};

    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.channel   = 'all';
    cfg.zlim      = 'maxabs';
    cfg.xlim      = [0 8];
    cfg.ylim      = [4 40];

    figure;
    ft_singleplotTFR(cfg, freqAvg);
    hold on;
    h = colorbar;
    ylabel(h, 'Z-scored Power (relative to 1–1.5s)', 'FontSize', 11);
    uniqueSubjCount = numel(unique(allSubjects));
    title({[ROI{1} ' All Subjects – ' ClassLookup{L}], ...
           [num2str(length(allChans)) ' chans across ' num2str(uniqueSubjCount) ' subjects']});
    % Markers
    xline(2,   '--k', 'Encoding',     'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom', 'FontSize', 10);
    xline(3.5, '--k', 'Delay Period', 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom', 'FontSize', 10);
    hold off;
end