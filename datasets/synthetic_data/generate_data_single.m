% -------------------------------------------------------------------------
% Communications and Signal Processing Group
% Department of Electrical and Electronic Engineering
% Imperial College London, 2023
%
% Supervisor  : Prof. Pier Luigi Dragotti
% Authors     : Vincent C. H. Leung
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%% Generating Training Data (Single Iteration) %%%%%%%%%%%%%%%%%%%
function [t_k,a_k,y_n]= generate_data_single(N,K_range,T,T_s,phi,samp_mode,PERIODIC,a_k,t_k)

%% Initialisation
% Parameters
[~,n1,n2]=n_vec(N);
t1 = n1 * T;
t2 = (n2+1) * T - T_s;
t = t1:T_s:t2;

if isscalar(K_range)
    K = K_range;
else
    K = randsample(min(K_range):max(K_range),1);
end

%% Generate Stream of Diracs
SAMELOC = 1;

while SAMELOC
    if nargin <= 8
        t_start=n1 * T;t_end=(n2+1) * T;

        % Randomly generation of location
        t_k = t_start + (t_end-t_start).*rand(K,1);

        %Sort the locations
        t_k = sort(t_k);
    end
    r=floor((t_k-t(1))/(t(2)-t(1))+1);
    r_uni = unique(r);
    SAMELOC = ~isequal(r, r_uni);
end
t_k=t(r);

if nargin <= 7
    a_k = 0.5+9.5*rand(K,1);      % Uniformly distributed amplitude from 0.5 to 10
end
x = zeros(size(t));
x(r)=a_k;

if K < max(K_range)
    t_k = [t_k(:); ones(max(K_range)-K,1)];
    a_k = [a_k(:); zeros(max(K_range)-K,1)];
end

if PERIODIC
    y = cconv(x,flip(phi),length(x));
    switch samp_mode
        case 'anticausal'
            y = circshift(y,-length(phi)+1);
        case 'causal'
            y = y;
        case 'peak'
            [~,idx] = max(abs(phi));
            shift = length(phi)-idx;
            y = circshift(y,-shift);
        otherwise
            error('Wrong sampling meode')
    end
else
    y = conv(x,flip(phi));
    switch samp_mode
        case 'anticausal'
            y = y(end-length(t)+1:end);
        case 'causal'
            y = y(1:length(t));
        case 'peak'
            [~,idx] = max(abs(phi));
            y = y(end-idx-length(t)+2:end-idx+1);
        otherwise
            error('Wrong sampling meode')
    end
end

% figure;
% plot(t,x); hold on
% plot(t,y);

idx = (0:N-1)/T_s*T+1;
y_n = y(idx);

end