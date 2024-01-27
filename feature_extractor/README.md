# The process of feature extraction

## Step 0: data preparation

### Download the VCTK Corpus

### Download the pretrained wav2vec 2.0 checkpoint

download the checkpoint on https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

## Step 1: audio resampling

Utilizing ffmpeg to resample the VCTK audio from 48k to 16k.

```
python renormalize.py
```

## Step 2: extract wav2vec 2.0 features

```
python w2vextractor.py
```

## Step 3: train the K-means model

Following the code in https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit, we train a K-means model for discrete token extraction.
```
python train_KMeans.py
```

## Step 4: extract discrete token features

```
python extract_tokens.py
```
