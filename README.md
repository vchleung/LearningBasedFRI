# [Learning-based Reconstruction of FRI Signals](https://arxiv.org/abs/2212.08758)
This repository is contains the code and pretrained models for "Learning-based reconstruction of FRI signals" by [Vincent C. H. Leung](https://www.imperial.ac.uk/people/chi.leung14) ([chi.leung14@imperial.ac.uk](mailto:chi.leung14@imperial.ac.uk)), [Jun-Jie Huang](https://jjhuangcs.github.io/), [Yonina C. Eldar](https://www.weizmann.ac.il/math/yonina/) and [Pier Luigi Dragotti](https://www.commsp.ee.ic.ac.uk/~pld/). 

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
Two types of data are used in the paper: synthetic and calcium imaging (cai-1) dataset. Please refer to [`datasets`](datasets) for related MATLAB codes and detailed instructions on how to use them.

## Reference
If you find this repository, please cite the following paper:

>V. C. H. Leung, J.-J. Huang, Y. C. Eldar and P. L. Dragotti. '[Learning-based
reconstruction of FRI signals](https://arxiv.org/abs/2212.08758),' Accepted to IEEE Transactions on Signal Processing, 2023.

```bibtex
@online{Leung2022,
  title = {Learning-based Reconstruction of {{FRI}} Signals},
  author = {Leung, Vincent C H and Huang, Jun-Jie and Eldar, Yonina C and Dragotti, Pier Luigi},
  date = {2023},
  pubstate = {Accepted to IEEE Transactions on Signal Processing},
  doi = {10.1109/TSP.2023.3290355}
}
```