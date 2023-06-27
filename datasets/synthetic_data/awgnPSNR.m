function out = awgnPSNR(in, psnr, a_k, realised)
% Add noise according to PSNR for a batch of data
% in is the input signal of size (num_data,N)
% psnr can be a scalar of a vector of size num_data
% a_k is of size (num_data,K)
% realised is bool to decide whether the noise power is the expected or
% realised

if nargin <= 3
    realised = false;
end

num_data = size(in,1);
N = size(in,2);
Pv2 = max(abs(a_k), [], 2).^2;
noisePower = 10.^(-psnr/10) .* Pv2;

if isreal(in)
    e     = randn(num_data, N) ;
else
    e     = randn(num_data, N) + 1j * randn(num_data, N);
    e     = sqrt(1/2) * e;
end

if realised
    P_e    = sum(e .* conj(e), 2) / N;
else
    P_e = 1;
end
e     = sqrt(noisePower./P_e) .* e;

out   = in + e;

end