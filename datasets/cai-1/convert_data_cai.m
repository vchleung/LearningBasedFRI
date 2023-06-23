% -------------------------------------------------------------------------
% Communications and Signal Processing Group
% Department of Electrical and Electronic Engineering
% Imperial College London, 2023
%
% Code to convert cai-1 GCaMP6f dataset into data within the settings of
% FRIED-Net, All the settings are as described in the paper
% 
% Adapted from plot_spikes_ca_traces.m by Tsai-Wen Chen, 2015/01/27 
% (Chen et. al. 2013 Nature; Akerboom, Chen 2012 J. Neurosci)
%
% Supervisor  : Prof. Pier Luigi Dragotti
% Authors     : Vincent C. H. Leung
% -------------------------------------------------------------------------

clear all
close all

N_list = [128, 64, 32, 32, 16]; % number of samples N
K_list = [7, 7, 7, 1, 1];       % number of spikes/pulses included in the new data
n_step  = 1;                    % window moving step, usually 1 to increase the effective amount of data
T_list = 1./N_list;              % sampling period of the OUTPUT locations, default = 1/N
train_test_split = 0.9;         % train-test split from the datase (assumed each mat contain signal of same length)
tol = 1e-4;                     % sanity check for the sampling period of the data
replacespike = 1;               % If the window contains less spikes than K, we fill the remaining with replacespike

% Flags to choose the conversion settings
RAW = 0;                        % raw data without neuropil correction
NORMALIZE = 0;                  % normalise the ENTIRE data stream such that the peak is 1 (NOT per window)
REMOVE_BIAS = 1;                % shift the signal by the lowest value such that every sample is non-negative and the smallest sample is 0
REMOVE_MEAN = 0;                % remove the mean  
INCLUDENOSPIKE  = 0;            % include windows that contain no spikes
SHUFFLE = 0;                    % shuffle the files to get random order of data

data_dir = "../dataset/cai-1/GCaMP6f_11cells_Chen2013/processed_data";  % Location of the calcium imaging data
cell="4";                       % specify which cell you want to focus on, "*" for all cells, default = 4
output_dir = 'cai-1/';
traintest_filenames = ["cai-train-N%dK%d.h5", "cai-test-N%dK%d.h5"];

data_filelist=dir(fullfile(data_dir, sprintf('data*_cell%s_*.mat', cell)));

train_data_size = floor(train_test_split * length(data_filelist));
if SHUFFLE
    data_filelist = datasample(data_filelist,length(data_filelist),'Replace',false);
end

train_data_filelist = data_filelist(1:train_data_size);
test_data_filelist = data_filelist(train_data_size+1:end);

%%
if ~isequal(length(N_list), length(K_list))
    error('Wrong Dimensions')
end
data_filelist = {train_data_filelist, test_data_filelist};

for z = 1:length(N_list)
    N = N_list(z);
    K = K_list(z);
    T_new = T_list(z);

    for i = 1:length(data_filelist) 
        dataIdx = 1;                    % initialise the data index 
        K_tot = [];
        y_n = [];
        t_k = [];
    
        filelist = data_filelist{i};    % train/test data
        for j = 1:length(filelist)
            filestruct = filelist(j);
            
            load(fullfile(filestruct.folder, filestruct.name));
            fmean_roi=obj.timeSeriesArrayHash.value{1}.valueMatrix;
            fmean_neuropil=obj.timeSeriesArrayHash.value{2}.valueMatrix;
            
            if RAW
                fmean_comp=fmean_roi;
            else
                fmean_comp=fmean_roi-0.7*fmean_neuropil;        % 0.7 given by original code
            end
            
            t_frame=obj.timeSeriesArrayHash.value{1}.time;
            filt=obj.timeSeriesArrayHash.value{4}.valueMatrix;
            t_ephys=obj.timeSeriesArrayHash.value{4}.time;
            
            detected_spikes=obj.timeSeriesArrayHash.value{5}.valueMatrix;
            spike_time=t_ephys(detected_spikes);
            
            sp = spike_time;
            t = t_frame;
            f = fmean_comp - min(fmean_comp);
            if NORMALIZE
                f = f/max(f);
            end
            
            if exist('T','var') && abs(T - mean(diff(t))) > tol
               warning('different sampling period')
            end
                
            T = mean(diff(t));
        
            %% Plot the original calcium imaging signals and ephys
            if z == 1
                figure;
                h1=subplot(2,1,1);
                plot(t_frame,f)
                title(filestruct.name,'Interpreter','none')
        
                h2=subplot(2,1,2);
                plot(t_ephys,filt,'r')
                hold on;
                plot(spike_time,filt(detected_spikes),'.k');
                linkaxes([h1,h2],'x');
            end
            %%
            n_start = 1;
    
            while n_start <= length(f) - N + 1      % Start from beginning of data and use sliding window
                t_range = t(n_start:n_start+N-1);
                sp_in_range = sp(sp>(min(t_range)-T) & sp<max(t_range));    % Count the spikes that are within the window
                K_window = min(K,length(sp_in_range));
                if ~isempty(sp_in_range) || INCLUDENOSPIKE           % Only takes the window if there is a spike unless it is test data or INCLUDENOSPIKE is asserted
                    K_tot(dataIdx) = length(sp_in_range);
                    y_n(dataIdx,:) = f(n_start:n_start+N-1);
    
                    if REMOVE_BIAS
                        y_n(dataIdx,:) = y_n(dataIdx,:) - min(y_n(dataIdx,:));
                    end
                    if REMOVE_MEAN
                        y_n(dataIdx,:) = y_n(dataIdx,:) - mean(y_n(dataIdx,:));
                    end
    
                    t_k(dataIdx,:) = ones(1,K) * replacespike;   
                    t_k(dataIdx,1:K_window) = (sp_in_range(1:K_window) - t_range(floor(N/2+1)))/T*T_new;
                    dataIdx = dataIdx + 1;
                end
                n_start = n_start + n_step;
            end
        
        end
        K_avg = mean(K_tot);
        K_max = max(K_tot);
        
        num_data = size(t_k,1);
        
        mkdir(output_dir)
        savepath = fullfile(output_dir,sprintf(traintest_filenames(i),N,K));
        
        h5create(savepath,'/y_n_noisy',[N num_data],'Datatype','single')
        h5write(savepath,'/y_n_noisy',single(y_n).')
        h5create(savepath,'/t_k',[K num_data],'Datatype','single')
        h5write(savepath,'/t_k',single(t_k).')
        h5create(savepath,'/filename',[1 length(filelist)],'Datatype','string')
        h5write(savepath,'/filename',string(vertcat(filelist.name)).')
        h5disp(savepath);
    
    end 
end
