function [phi,t_phi] = eMOMS(P, resolution)

t_phi = 0:1/resolution:(P+1)-1/resolution;
t_phi = t_phi(:);
lambda = 2*pi/(P+1); 
phi = 1;
for i = 1:P/2
    phi = phi + 2*cos(i*lambda*(t_phi-P/2));
end
phi = phi(:)./(P+1);

end