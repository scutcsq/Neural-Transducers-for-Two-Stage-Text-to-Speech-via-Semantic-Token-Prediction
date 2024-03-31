A code version with reference to vall-e and k2 code

# 1、Step 1 Data Preprocess

Download the code from https://github.com/lifeiteng/vall-e, and replace the code(tokenizer.py) in valle.
Follow the processing process in vall-e.

```
cp tokenizer.py valle/egs/libritts/bin/tokenizer.py
cd valle/egs/libritts
```
put the faiss token scp file in here
```
cp xxx.ark xxx.scp ./
```
put the 24k mel scp file in here
```
cp mels.ark mels.scp ./
```
run preprocess pipeline
```
bash prepare.sh --stage -1 --stop-stage 3
```

# 2、Step 2 Run Stage 1 Model
change the workspace
```
cd dynamic_version
```
create the data soft link to the workspace
```
ln -s valle/egs/libritts/data data
```
run the stage 1 model
```
CUDA_VISIBLE_DEVICES=0 python train_dynamic.py --max-duration 240 --filter-min-duration 0.5 --filter-max-duration 14 --num-buckets 6 --save-every-n 10000
```
