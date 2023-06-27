# Learning-based Reconstruction of FRI Signals
This repository is contains the code and pretrained models for [Learning-based reconstruction of FRI signals](https://arxiv.org/abs/2212.08758) by [Vincent C. H. Leung](https://www.imperial.ac.uk/people/chi.leung14), Jun-Jie Huang, Yonina C. Eldar and Pier Luigi Dragotti. 

The code consists of two different learning-based FRI models: Deep Unfolded Projected Wirtinger Gradient Descent (Deep Unfolded PWGD) and FRI Encoder-Decoder Network (FRIED-Net). Deep Unfolded PWGD operates in the frequency domain and unfolds the iterative denoising process, while FRIED-Net models the FRI acquisition process and the fact that FRI signals are determined by a small number of parameters to form an autoencoder-like network architecture. 

## Prerequisites
The code was tested on Python 3.10 so this particular version is recommended. 

The code for both models was developed based on PyTorch. `torch>=2.0.0` is required specifically for Deep Unfolded PWGD since it deals with complex numbers. You can install the latest version of PyTorch [here](https://pytorch.org/get-started/locally/).

Other required packages are listed in [`requirements.txt`](requirements.txt).

## Deep Unfolded Projected Wirtinger Gradient Descent (Deep Unfolded PWGD)
Please refer to [`DeepUnfoldedPWGD`](DeepUnfoldedPWGD) for the implementation and pretrained models. 

## FRI Encoder-Decoder Network (FRIED-Net)
Please refer to [`FRIED-Net`](FRIED-Net) for the implementation and pretrained models.

## Dataset
Two kinds of data are used in the paper: simulated and calcium imaging (cai-1) dataset.

For simulated dataset, 

We have also provided the calcium imaging related MATLAB code under [`datasets/cai-1`](datasets/cai-1), where you can convert the cai-1 dataset into the FRI settings used by FRIED-Net, as well as plotting the reconstruction results. Please download the original cai-1 dataset [here](https://crcns.org/data-sets/methods/cai-1) before running the code. In addition, we have provided the code for running fast deconvolution based on [PyFNND](https://github.com/alimuldal/PyFNND) for performance comparison. Our pretrained FRIED-Net models can be found in [`FRIED-Net/pretrained_models`](FRIED-Net/pretrained_models)

## Reference
If you find this repository, please cite the following paper

>V. C. H. Leung, J.-J. Huang, Y. C. Eldar and P. L. Dragotti. '[Learning-based
reconstruction of FRI signals](https://arxiv.org/abs/2212.08758),' Accepted to IEEE Transactions on Signal Processing, 2022.

```bibtex
@online{Leung2022,
  title = {Learning-Based Reconstruction of {{FRI}} Signals},
  author = {Leung, Vincent C H and Huang, Jun-Jie and Eldar, Yonina C and Dragotti, Pier Luigi},
  date = {2023},
  pubstate = {Accepted to IEEE Transactions on Signal Processing},
  doi = {10.1109/TSP.2023.3290355}
}
```