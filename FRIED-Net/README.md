# FRI Encoder-Decoder Network (FRIED-Net)
An encoder-decoder network inspired by modelling the acquisition process of FRI signals as it can be perfectly defined by a small number of parameters. We demonstrate the common case of reconstructing a stream of Diracs/pulses, which is completely defined by its locations (latent code of the encoder-decoder network) and amplitudes (obtained through least squares fitting).

<p align="center">
    <img src="../figures/FRIED-Net.png" width="70%">
</p>

## Data
Data is in `.h5` format and must at least contain the noisy samples as `y_n_noisy` and the ground truth locations `t_k`. 

## Training
First, start the ``visdom.server`` for visualisation during training.
```shell
python -m visdom.server
```
To start training, run[`main.py`](main.py). The code allows training in different settings and below are some examples of its usage in reconstructing periodic streams of Diracs (`Number of Samples N=21, Number of Diracs K=2, Sampling Period T=1/21`). For convenience, the arguments that are closely related are grouped together and shown on the same line.

To train a direct inference model (encoder only):
```shell
python main.py \
    --output_dir path/to/output/ --data_dir path/to/data/ --N 21 --K 2 \
    --model_encoder conv \
    --loss_fn tkMSE
```
Below trains a full FRIED-Net with encoder initialised from the pretrained direct inference model, with the decoder fixed to be linearly approximated eMOMS of order `P=20`.  
```shell
python main.py \
    --output_dir path/to/output/ --data_dir path/to/data/ --N 21 --K 2 \
    --model_encoder conv --encoder_init_path path/to/directInferenceModel.pth \
    --model_decoder decoderReLUNet --init_phi eMOMS --init_phi_prms 20 --true_ak \
    --loss_fn ynMSE+tkMSE --loss_prms 1.
```
When the shape of $\varphi(t)$ is unknown and we want to learn the decoder 
```shell
python main.py \
    --output_dir path/to/output/ --data_dir path/to/data/ --N 21 --K 2 \
    --model_encoder conv --encoder_init_path path/to/directInferenceModel.pth --no-train_encoder \
    --model_decoder decoderReLUNet --train_decoder --norm_phi \
    --loss_fn ynnoisyMSE 
```
Finally, when you want to initialise from the learned encoder and learned decoder to do further training:
```shell
python main.py \
    --output_dir path/to/output/ --data_dir path/to/data/ --N 21 --K 2 \
    --model_encoder conv --encoder_init_path path/to/directInferenceModel.pth --train_encoder \
    --model_decoder decoderReLUNet --decoder_init_path path/to/learnedPhiModel.pth --train_decoder --norm_phi \
    --loss_fn ynnoisyMSE+tkMSE --loss_prms 100.
```
There are also other different settings to be played with. For example, to apply FRIED-Net to Calcium imaging data (non-periodic signal), arguments `--no-periodic` and `--samp_mode causal` has to be set. Please refer to the description of the arguments in the code for a more detailed explanation to each parameter.

## Testing
To test the trained model, run [`test.py`](test.py):
```shell
python test.py \
    --data_dir path/to/data/ \
    --data_filename filenames \
    --output_dir path/to/output/ \
    --model_path path/to/model_last.pth
```
To fine-tune the learned encoder using backpropagation of mean squared samples error, set the flag `--fine_tune` and `--batch_size 1`. It is also possible to evaluate multiple datafiles using a single model or evaluate multiple datafile with a specified model for each of them by passing `--data_filename` or `--model_path` a list. For testing on calcium imaging data, set the flag `--load_cai-mat` and point `--data_filename` to the raw cai-1 recording. 