# Efficient Neural Architecture Search (ENAS)

This repo contains a PyTorch implementation of [Efficient Neural Architecture Search via Parameters Sharing](https://arxiv.org/abs/1802.03268).

This implementation is a port of the [official Tensorflow implementation](https://github.com/melodyguan/enas). As such, I have tried to replicate hyperparameter settings and "secret sauce" tricks of the original implementation as closely as possible, although there still appears to be some differences in performance.

Currently only the CNN macro architecture search has been implemented. For a PyTorch implementation of RNN cell search see [carpedm20's ENAS repo](https://github.com/carpedm20/ENAS-pytorch).