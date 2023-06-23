from pyfnnd import deconvolve, demo, plotting
import matplotlib.pyplot as plt
import scipy.io as sio
from os.path import join as pjoin
import os

data_dir = "dataset/cai-1/GCaMP6f_11cells_Chen2013/processed_data/"
test_filename = "data_20120521_cell4_007.mat"
out_dirname = pjoin(data_dir, 'results', 'fast_deconv')

contents = sio.loadmat(pjoin(data_dir, test_filename))

fmean_roi = contents['obj']['timeSeriesArrayHash'][0][0][0][0]['value'][0][0][0][0]['valueMatrix']
fmean_neuropil = contents['obj']['timeSeriesArrayHash'][0][0][0][0]['value'][0][1][0][0]['valueMatrix']
t = contents['obj']['timeSeriesArrayHash'][0][0][0][0]['value'][0][0][0][0]['time']
dt = float(t[1]-t[0])
F = fmean_roi
F = F - 0.7 * fmean_neuropil    #Neuropil correction
F = F.T

# deconvolve it, learning alpha, beta and lambda
n_best, C_best, LL, theta_best = deconvolve(
    F, dt=dt, verbosity=1, learn_theta=(0, 1, 1, 1, 0),
    spikes_tol=1E-6, params_tol=1E-6
)

print(len(n_best))

fig, axes, F_hat = plotting.plot_fit(F, n_best, C_best, theta_best, dt=dt)
fig.set_size_inches(12, 6)
fig.tight_layout()

if not os.path.exists(out_dirname):
    os.makedirs(out_dirname)
    print("Directory", out_dirname, "Created ")

sio.savemat(pjoin(out_dirname,test_filename), {'hist': n_best, 'y_n_est_batch': F_hat})

plt.show()
