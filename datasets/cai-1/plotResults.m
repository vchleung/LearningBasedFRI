% -------------------------------------------------------------------------
% Communications and Signal Processing Group
% Department of Electrical and Electronic Engineering
% Imperial College London, 2023
%
% Supervisor  : Dr Pier Luigi Dragotti
% Author      : Vincent Leung, Modified from Jon Onativia
%
% File        : plotResults.m
% -------------------------------------------------------------------------
% Ca transient detection with double consistency (from multiple FRIED-Net
% models)
% -> big window and K estimation
% -> small window and fixed K
% Spikes detected from the union of the histograms

clc
clear all
close all

%%
raw_data_path = "../../dataset/cai-1/GCaMP6f_11cells_Chen2013/processed_data/data_20120521_cell4_007.mat"; % path to the original calcium imaging data

results_dir = [
    "../../FRIED-Net/results/cai-1", ...
    "../../results/cai-1/fast_deconv", ...    
    ];

filenames = {
    ["N128K7","N64K7","N32K7","N32K1","N16K1"], ... 
    "fast_deconv", ...
    };

method_names = [
    "FRIED-Net", ...
    "Fast Deconv", ...
    ];                           % For titles in plots

hist_fac = {
    [1, 1, 1, 2, 1], ...
    NaN, ...
    };                           % Multiplication factor for each histogram


threshold_range = 0:0.002:0.4;   % The range of threshold (normalised, [0,1]) to plot ROC curve
T_a_fac = 2;                        % T_a = T_a_fac*T, If a spike is found between the acceptance interval (t-T_a, t+T_a), it is a true positive

axis_size = 10;
label_size = 12;

%% Loading ground truth data
load(raw_data_path)
fmean_roi=obj.timeSeriesArrayHash.value{1}.valueMatrix;
fmean_neuropil=obj.timeSeriesArrayHash.value{2}.valueMatrix;
fmean_comp=fmean_roi-0.7*fmean_neuropil;
noisy_signal = fmean_comp - min(fmean_comp);    % Neuropil correction from plots
t=obj.timeSeriesArrayHash.value{1}.time;
t=t(1:length(noisy_signal));                    % match the length of t with the noisy signal
filt=obj.timeSeriesArrayHash.value{4}.valueMatrix;
t_ephys=obj.timeSeriesArrayHash.value{4}.time;
detected_spikes=obj.timeSeriesArrayHash.value{5}.valueMatrix;
sp=t_ephys(detected_spikes);

T_cai = mean(diff(t));                          % Sampling period in the raw data
hist_t     = t;
hist_len   = length(hist_t);

%% Initialising parameters
if ~isequal(length(results_dir), length(filenames), length(method_names), length(hist_fac))
    error('Wrong Dimensions')
end

hit_rate  = zeros(length(method_names),length(threshold_range));
false_pos = zeros(length(method_names),length(threshold_range));
mse       = zeros(length(method_names),length(threshold_range));
detectedsp_tot = zeros(length(method_names),length(threshold_range));
true_pos = zeros(length(method_names),length(threshold_range));

%% Main Loop (Loading results from models)
for m = 1:length(method_names)
    %% Load results
    if length(filenames{m}) == 1
        load(fullfile(results_dir(m), filenames{m}), "hist")
        hist_sp = hist.';
        max_detect = max(hist_sp);
        clear t_k sspp_w N_list K_list
    else
        num_models = length(filenames{m});
        hist_sp_w = zeros(hist_len, num_models);
        sspp_w = zeros(hist_len, num_models);
        N_list = zeros(1, num_models);
        K_list = zeros(1, num_models);

        for n = 1:num_models
            % Load results from models
            load(fullfile(results_dir(m), filenames{m}(n)), "t_k_est", "N", "K", "T")
            N = double(N);
            N_list(n) = N;
            K_list(n) = K;

            % Transform from the windowed signals/locations to the real
            % signal (Reverse the sliding window), any locations outside of
            % the window are rejected
            t_k_est_adjusted = zeros(size(t_k_est));
            for i =1:size(t_k_est,1)
                tmp = t_k_est(i,:)/T*T_cai+t(i+floor(N/2+1));
                tmp(tmp<t(i)-T_cai)=NaN;
                tmp(tmp>t(i+N-1))=NaN;
                t_k_est_adjusted(i,:) = tmp;
            end
            t_k = t_k_est_adjusted(~isnan(t_k_est_adjusted));

            % Count the number of detected exponentials within a time interval
            for ith_t = 1 : hist_len
                t_i = hist_t(ith_t);                   

                inds           = find(t_k > (t_i - T_cai/2) & t_k < (t_i + T_cai/2)); 
                hist_sp_w(ith_t,n) = length(inds);

%                 sspp_w(ith_t,n) = t_i;            % the reconstructed location is the center of the time step t_i           
                sspp_w(ith_t,n) = mean(t_k(inds));  % the reconstructed location is the mean from all the sliding windows of a single model
            end
        end
        hist_sp = sum(hist_sp_w.*hist_fac{m},2);
        max_detect = sum(N_list.*hist_fac{m},2);
    end

    %% Find the peaks of histogram
    for k = 1:length(threshold_range)
        threshold_fac = threshold_range(k);

        % Only take into account the peak of the histograms
        sspp       = [];
        sspp_tidx   = [];
        threshold  = threshold_fac * max_detect;    % unnormalised threshold
        for ith_t = 1 : hist_len
            if hist_sp(ith_t) > threshold ...
                    &&  (ith_t < hist_len && (hist_sp(ith_t) >= hist_sp(ith_t+1)) ) ...
                    && (ith_t > 1 && (hist_sp(ith_t) >  hist_sp(ith_t-1)) )  % if it is higher than threshold and it is a peak
                t_i  = hist_t(ith_t);

                if exist('t_k', 'var') && exist('sspp_w', 'var')    % if there are multiple models
                    inds = find(t_k > (t_i - T_cai/2) & t_k < (t_i + T_cai/2));

                    % calculate the weighted mean of the locations recovered by mutiple models
                    tmpp = sspp_w(ith_t,:).*hist_fac{m}.*hist_sp_w(ith_t,:);    
                    tmpp_idx = ~isnan(tmpp);
                    sspp = [sspp; sum(tmpp(tmpp_idx))/hist_sp(ith_t)];
                else
                    sspp = [sspp; t_i];
                end
                sspp_tidx = [sspp_tidx; ith_t];
            end
        end

        sspp_peak = sspp;

        %%
        % Compare the detected spikes with the real spikes
        sp = sp(sp > t(1) & sp<t(end));
        num_sp   = length(sp);
        hit_sp   = false(num_sp, 1);
        sspp_ids = [];
        sspp_false_pos = sspp;
        T_a  = T_a_fac*T_cai;
        for ith_sp = 1 : num_sp
            t_i  = sp(ith_sp);
            inds = find(sspp_false_pos > (t_i - T_a) & sspp_false_pos < (t_i + T_a));

            if ~isempty(inds)
                hit_sp(ith_sp) = true;
                sspp_ids       = [sspp_ids; find(sspp == sspp_false_pos(inds(1)))];

                % Remove this spike from detected spikes
                sspp_false_pos(inds(1)) = [];
            end
        end

        %% Plot an example histogram for each method
        data_idx = 10000;
        plot_length = 4000;

        if threshold_fac == 0.1
            figure
            set(gcf, 'Position', [450+200*m 300 900 300])
            gt = stem(sp, max_detect*ones(size(sp)), 'Linewidth', 0.9, 'Marker', 'none'); hold on
            plot(hist_t, hist_sp, 'k', 'Linewidth', 1);
            yline(threshold, 'k--');
            est = scatter(hist_t(sspp_tidx), hist_sp(sspp_tidx), 'r');
            tp = scatter(hist_t(sspp_tidx(sspp_ids)), hist_sp(sspp_tidx(sspp_ids)), 'rx');
            xlim([t(data_idx), t(data_idx+plot_length)]);
            ylim([0 max_detect]);
            title(method_names(m)+sprintf(', threshold = %.3f, $T_a$ = %dT', threshold_fac, T_a_fac), 'Interpreter', 'latex');
            hdl = legend([gt, est, tp], ["Ground Truth", "Reconstructed", "True Positive"], 'Location', 'Northeast');
            set(hdl, 'FontSize', label_size)
            set(gca, 'FontSize', axis_size)
        end

        if threshold_fac == 0.1 && m==1
            aplot = 100;
            figure
            set(gcf, 'Position', [650 300 650 220])
            y = plot(t, noisy_signal, 'k', 'Linewidth', 1); hold on
            gt = stem(sp, aplot*ones(size(sp)), 'filled', '^b', 'Linewidth', 1);
            est = stem(hist_t(sspp_tidx), 0.975*aplot*ones(size(sspp_tidx)), 'r', 'Linewidth', 1);
            xlim([30, 60]);
            ylim([0 2.2*aplot]);
            xlabel('Time (s)', 'FontSize', axis_size, 'Interpreter', 'Latex')
            hdl = legend([y, gt, est], ["Fluorescent Signal", "Ground Truth", "Reconstructed"], 'Location', 'Northwest', 'Interpreter', 'Latex');
            set(hdl, 'FontSize', label_size)
            set(gca, 'FontSize', axis_size)
        end

        %% Accuracy of detected spikes
        hit_rate(m,k)  = sum(hit_sp) / length(hit_sp) * 100;
        true_pos(m,k) = sum(hit_sp);
        false_pos(m,k) = length(sspp_false_pos);
        mse(m,k)    = mean((sp(hit_sp) - sspp(sspp_ids)).^2);
        detectedsp_tot(m,k) = length(sspp);


    end

    %% Print the results
    disp('')
    disp('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    disp('++++ SLIDING WINDOW Ca transient detection algorithm double consistency')
    if exist("N_list")
        disp(sprintf('++++ %s, T_a = %dT, N = %s, K = %s', method_names(m), T_a_fac, num2str(N_list,'%d '), num2str(K_list,'%d ')))
    else
        disp(sprintf('++++ %s, T_a = %dT', method_names(m), T_a_fac))
    end
    disp(['Threshold                       : ' num2str(threshold_range, '%-14.3f')])
    disp(['Total number of real spikes     : ' num2str(num_sp)])
    disp(['Total number of detected spikes : ' num2str(detectedsp_tot(m,:), '%-14i')])
    disp(['Real spikes detected            : ' num2str(true_pos(m,:), '%-14i')])
    disp(['MSE of spike locations          : ' num2str(mse(m,:), '%-14.4f')])
    disp(['RMSE of spike locations         : ' num2str(sqrt(mse(m,:)), '%-14.4f')])
    disp(['Spike detection rate            : ' num2str(hit_rate(m,:),'%11.1f %%')])
    disp(['False positives                 : ' num2str(false_pos(m,:), '%-14i')])
    disp(['False positives rate            : ' num2str(false_pos(m,:)/(t(end)-t(1)), '%11.4f Hz')])
    disp(' ')

end


%% Plot ROC curve
if length(threshold_range) ~= 1
    linS = ["-","-."];
    figure;
    set(gcf, 'Position', [50 300 275 220])

    for m = 1:length(method_names)
        plot3(squeeze(false_pos(m,:))/(t(end)-t(1)), squeeze(hit_rate(m,:)), threshold_range, linS(m), 'Linewidth', 2); hold on
    end

    grid minor
    xlabel('False Positive Rate (Hz)', 'FontSize', label_size, 'Interpreter', 'Latex')
    ylabel('True Positive Rate (\%)', 'FontSize', label_size, 'Interpreter', 'Latex')
    hdl = legend(method_names, 'Location','southeast', 'Interpreter', 'Latex');
    set(hdl, 'FontSize', label_size)
    set(gca, 'FontSize', axis_size)
    title(sprintf("ROC Curve, $T_a$ = %dT",T_a_fac),'Interpreter','latex')
    ylim([0 100]);
    xlim([0 1]);
    view(0,90)

    figure;
    set(gcf, 'Position', [50 300 200 220])
    h = boxplot(squeeze(sqrt(mse)).', 'Labels', method_names);
    set(h,{'linew'},{1})
    ylabel('SD (s)','Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
    set(gca, 'FontSize', axis_size)
    ax = gca;
    ax.YAxis.Exponent = -2;
end
