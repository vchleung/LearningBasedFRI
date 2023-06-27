function [n_vec,n1,n2]=n_vec(N)

if mod(N, 2) == 0
    n1 = -N/2;
    n2 = N/2 - 1;
else
    n1 = -(N-1)/2;
    n2 = (N-1)/2;
end
n_vec = (n1:n2)';

end