# Datasets
This folder contains the MATLAB codes related to the two datasets used in the paper: a synthetic streams of pulses dataset and cai-1 calcium imaging dataset. The data used in the paper were generated using the default settings fixed in the code.

## Synthetic dataset
The simulated dataset comprises streams of Diracs sampled by sampling kernel $varphi(t)$. 

The main script is [`generate_data.m`](synthetic_data/generate_data.m). Running the script generates the training and testing data where the locations are uniformly distributed across the time range, and amplitudes are uniformly distributed between 0.5 and 10. It supports the choice of sampling kernel being eMOMS and E-Spline, as well as some other settings such as PSNR, etc. Since Deep Unfolded PWGD requires transforming FRI signals into sum of exponentials (frequency domain), the code also offers the choice of saving frequency related parameters.

A notable case that was looked into was when we change the distance between the Diracs/pulses when $K=2$. Therefore, [`generate_data_K2_fixeddeltat.m`](synthetic_data/generate_data_K2_fixeddeltat.m) generates the evaluation dataset where the locations of the Diracs are fixed with certain distances, while amplitudes are fixed to be equal but again uniformly distributed between 0.5 and 10.

Some examples of the exact synthetic data used in the paper can be found [here](https://drive.google.com/drive/folders/1whT3xE-22EMH-KJBLu2k8XCUK7wECmVf?usp=sharing). 

The pretrained models can be found in [`FRIED-Net/pretrained_models/synthetic_data`](../FRIED-Net/pretrained_models/synthetic_data) and [`DeepUnfoldedPWGD/pretrained_models/synthetic_data`](../DeepUnfoldedPWGD/pretrained_models/synthetic_data).

## Calcium Imaging Dataset (cai-1)
Another application of our proposed FRIED-Net is on detecting spikes from calcium imaging recordings. In the paper, we have used the GCaMP6f data from the cai-1 dataset described in Chen, et al Nature 2013.

Under [`cai-1`](cai-1), we have provided the MATLAB code [`convert_data_cai.m`](cai-1/convert_data_cai.m) to convert the GCaMP6f calcium imaging recordings into the FRI settings used by FRIED-Net, as well as plotting the reconstruction results. Please first download the original GCaMP6f recordings from cai-1 dataset [here](https://crcns.org/data-sets/methods/cai-1) before running the code.

In addition, we have provided the code for running fast deconvolution based on [PyFNND](https://github.com/alimuldal/PyFNND) for performance comparison. Our pretrained FRIED-Net models can be found in [`FRIED-Net/pretrained_models/cai-1`](../FRIED-Net/pretrained_models/cai-1).
