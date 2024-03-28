# Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction

*Unofficial Pytorch reproduction* for the paper "Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction" (https://arxiv.org/pdf/2401.01498.pdf)

This project will be updated slowly.

I'm clutching my master's thesis, this project may be stopped for a month or two.
# To Do List

- [ ] Resolve the convergence issue in the first stage of model training
- [x] Dynamic Batch Training
- [x] Utilize the ConvNet for training
- [x] IPA Phoneset
- [x] Finish the second stage of model training
- [x] Finish the token extraction by K-means model
- [x] Reproduce the model on LibriTTS

## Device

1 A100 80GB

## Preparation

### Corpus

[VCTK](https://datashare.ed.ac.uk/handle/10283/2651)

[LibriTTS](https://www.openslr.org/60/)

### Features

[wav2vec2.0-XLSR](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)

### k2 installation

https://k2-fsa.github.io/k2/installation/index.html

## Citation

Kim M, Jeong M, Choi B J, et al. Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction[J]. arXiv preprint arXiv:2401.01498, 2024.

Kim M, Jeong M, Choi B J, et al. Transduce and speak: Neural transducer for text-to-speech with semantic token prediction[C]//2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023: 1-7.
