# Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction

*Unofficial Pytorch reproduction* for the paper "Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction" (https://arxiv.org/pdf/2401.01498.pdf)

This project will be updated slowly.

- [ ] Resolve the convergence issue in the first stage of model training
- [x] Finish the second stage of model training
- [x] Finish the token extraction by K-means model
- [ ] Reproduce the model on LibriTTS

## Device

1 A100 80GB

## Preparation

### Corpus

VCTK https://datashare.ed.ac.uk/handle/10283/2651

### Features

wav2vec2.0-XLSR (https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

### k2 installation

https://k2-fsa.github.io/k2/installation/index.html

## Citation

Kim M, Jeong M, Choi B J, et al. Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction[J]. arXiv preprint arXiv:2401.01498, 2024.
