% -------------------------------------------------------------------------
% Communications and Signal Processing Group
% Department of Electrical and Electronic Engineering
% Imperial College London, 2023
%
% Supervisor  : Prof. Pier Luigi Dragotti
% Authors     : Vincent C. H. Leung
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%% Generating Evaluation Data with fixed distance between pulses (Main Routine) %%%%%%%%%%%%%%%%%%%

% close all
clear all

%% Initialise Parameters
kernel        = "eMOMS";
P             = N-1;            % Order of sampling kernel

% kernel        = "ESpline";
% P             = 6;

N             = 21;             % number of temporal samples
K             = 2;              % Number of Diracs, must be 2 in this code
T             = 1/N;            % sampling period
resolution    = 64;             % how many points of phi within a sampling period T
samp_mode     = "anticausal";   % 'peak', 'causal', 'anticausal'
PERIODIC      = 1;              % 1 if the signal is periodic

t_k0          = 0.1;            % The location of the 1st dirac/pulse (fixed) 
% t_k0 = 0

% since FRIED-Net does not require c_mn or h or s[m], we can choose whether
% to save them in the dataset
% "": don't save any of c_mn or h or s[m], 
% "c": save c_mn and h
% "s": save s[m], \tilde{s}[m] and h
% "all": save c_mn, s[m], \tilde{s}[m], and h
SAVE_FREQ = "c";              

% Data settings
output_dir    = "../../dataset/fixeddeltat";
out_filename = "deltat%d_%sdB.h5";
num_data = 1e4;                 % Number of data
PSNR_range = 70:-5:-5;          % Select PSNR

%% Select sampling kernel
m = 0:P;
if strcmp(kernel,'eMOMS')
    [phi,t_phi] = eMOMS(P, resolution);
    alpha_0 = - 1j * pi/ (P+1) *P ;
    lambda  = 2j*  pi / (P+1) ;
    alpha_vec    = alpha_0 + lambda * m;
elseif strcmp(kernel,'ESpline')
    L = 3.5*(P+1);
    alpha_0 = - 1j * pi/ L *P ;
    lambda  = 2* 1j * pi / L ;
    alpha_vec    = alpha_0 + lambda * m;
    [phi,t_phi] = ESpline(alpha_vec, resolution);
else
    error("Wrong Kernel Selected")
end

% change phi according to different sampling mode
if strcmp(samp_mode,'peak')
    [~,idx] = max(abs(phi));
    t_phi = t_phi - t_phi(idx);
elseif strcmp(samp_mode, 'causal')
    phi = phi(end:-1:1);
    t_phi   = -t_phi(end:-1:1);
elseif ~strcmp(samp_mode, 'anticausal')
    error('Wrong sampling mode selected')
end


%% Initialise the locations of the pulses
T_s           = T/resolution;    % "continuous" time resolution

if K ~= 2
    error('K must be 2')
end

t_k_array = zeros(K,11);

[n_vec,n1,n2]=n_vec(N);
t_start    = n1 * T;
t_end    = (n2+1) * T - T_s;

t_k_array(1,:) = t_k0;
t_k_array(2,:) = t_k0 +logspace(-0.5,-3,11);

if any(t_k_array(2,:) > t_end)
    error('Not valid t_k0 or separation')
end

%% Plot sampling filter h(t) = phi(-t/T)
figure;
plot(-t_phi*T, phi);    % From sampling kernel to sampling filter, t axis change
title('Impulse response $h(t) = \varphi(-t/T)$','Interpreter','latex')

%% Find c_mn 
mkdir(output_dir)

if SAVE_FREQ ~= ""
    c_mn = get_c_mn_exp(alpha_vec, n_vec, phi, t_phi);
    if SAVE_FREQ == "c" || SAVE_FREQ == "all"
        save(fullfile(output_dir, "c_mn.mat"), "c_mn")
    end
end

%% Main Loop
for j = 1:size(t_k_array,2)
    inloc = t_k_array(:,j);
    for k = 1:length(PSNR_range)
        PSNR = PSNR_range(k);
        % Add noise to the samples
        y_n = zeros(num_data,N);
        t_k = zeros(num_data,K);
        a_k = zeros(num_data,K);

        for i =1:num_data
            inamp = (0.5+9.5*rand)*ones(K,1);  % Amplitudes are equal but uniformly distributed between 0.5 and 10
            [t_k(i,:),a_k(i,:),y_n(i,:)] = generate_data_single(N,K,T,T_s,phi,samp_mode,PERIODIC,inamp,inloc);
        end
        t_k = t_k(1,:);
        t_k_array(:,j) = t_k;

        y_n_noisy = awgnPSNR(y_n, PSNR, a_k);
        
        %% writing to HDF5
        savepath = fullfile(output_dir, sprintf(out_filename, j, num2str(PSNR)));
        
        h5create(savepath,'/y_n_noisy',[N num_data],'Datatype','single')
        h5write(savepath,'/y_n_noisy',single(y_n_noisy).')
        h5create(savepath,'/t_k',[K 1],'Datatype','single');
        h5write(savepath,'/t_k',single(t_k(:)));
        h5create(savepath,'/a_k',[K num_data],'Datatype','single')
        h5write(savepath,'/a_k',single(a_k).')
        h5create(savepath,'/y_n',[N num_data],'Datatype','single')
        h5write(savepath,'/y_n',single(y_n).')

        if SAVE_FREQ ~= ""
            if SAVE_FREQ == "s" || SAVE_FREQ == "all"
                s_m = y_n * c_mn.';
                h5create(savepath,'/s_m',[2*(P+1) num_data],'Datatype','single')
                h5write(savepath,'/s_m',single([real(s_m), imag(s_m)]).')
                s_m_noisy = y_n_noisy * c_mn.';
                h5create(savepath,'/s_m_noisy',[2*(P+1) num_data],'Datatype','single')
                h5write(savepath,'/s_m_noisy',single([real(s_m_noisy), imag(s_m_noisy)]).')
            end
        end

        h5disp(savepath);
                
    end
end
