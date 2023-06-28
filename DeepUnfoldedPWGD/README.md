# Deep Unfolded Projected Wirtinger Gradient Descent (Deep Unfolded PWGD)
A model-based deep learning method inspired by unrolling the iterative denoising method that is usually done before applying Prony's method to recover exponentials.

<p align="center">
    <img src="../figures/DeepUnfoldedPWGD.png" width="70%">
</p>

## Data
Data is in `.h5` format. Since this is operated in frequency domain, it requires either the exponential reproducing matrix `c_mn` to convert the time-domain samples, or the converted frequency samples `s[m]`. Using the annihilating filter loss would also require the ground truth annihilating filter `h`. These options can be customised during data generation in [`datasets`](../datasets). 

## Training
First, start the ``visdom.server`` for visualisation during training.
```shell
python -m visdom.server
```
To start training, run[`main.py`](main.py). For instance, to train an unfolded PWGD model using annihilating filter loss:
```shell
python main.py \
    --output_dir path/to/output/ --data_dir path/to/data/ --c_mn_filename c_mn.mat \
    --N 21 --P 20 --K 2 --model UnfoldedPWGD --P_rank_method svtK+1 \
    --loss_fn afLoss --loss_prms 10 5e-3
```
Note that `c_mn_filename` is optional. If unspecified, the code reads `s_m_noisy` and `s_m` directly from the .h5 file

## Testing
To test the trained model, run [`test.py`](test.py):
```shell
python test.py \
    --data_dir path/to/data/ --c_mn_filename c_mn.mat \
    --output_dir ./results/test/ --model_path .\test\afLoss\model_last.pth \
    --prony --L 21 
```
`--prony` means that the testing involves running Prony's method after estimating the denoised sum of exponentials $s[m]$ to eventually retrieve the locations $t_k$ in time domain. To do that, we have to also specify `L`, which represents the frequency separation $\lambda = \frac{2\pi}{L}$ and the sampling period $T$
