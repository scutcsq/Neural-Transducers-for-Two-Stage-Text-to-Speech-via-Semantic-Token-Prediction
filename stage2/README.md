# Stage 2: transform the semantic token to speech

## Step 0: data preparation

### copy the semantic token scp file and mel scp file to this folder
```
cp ../feature_extractor/mel16k.scp ./
cp ../feature_extractor/mel16k.ark ./
cp ../feature_extractor/token.scp ./
cp ../feature_extractor/token.ark ./ 
```

### clone the code of vits and rename the vits folder to 'vits'

## Step 1: train the model

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Step 2: inference the model

```
CUDA_VISIBLE_DEVICES=0 python inference.py --cp_path=ckpt_path
```

# Reference

the architecture of the codes is referenced from [VITS](https://github.com/jaywalnut310/vits)

the ref_enc.py code is referenced from [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
