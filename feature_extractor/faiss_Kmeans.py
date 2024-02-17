import sys
import os
import faiss
import numpy as np
import torch
import random
import kaldiio
in_dir = r'./wav2vec2_15th/'
names = os.listdir(in_dir)
random.shuffle(names)
totallens = len(names)
stoplens = int(0.3 * totallens) # If the space is large enough, you could utilize the total lengths
fea_batch = []
# Merge the features
count = 0
for name in names:
    dataname = os.path.join(in_dir, name)
    feat = np.load(dataname)
    fea_batch.append(feat[0, :, :])
    count += 1
    if count == stoplens:
        break
fea_batch = np.vstack(fea_batch)
print('fea_batch: ', fea_batch.shape)

# train the Kmeans model
ncentroids = 512
niter = 200
verbose = True
# d = 1024
d = fea_batch.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(fea_batch)

#extract features
faiss_k_means = {}
token_out_dir = r'./token_wav2vec2_15th/'
os.makedirs(token_out_dir, exist_ok= True)
for name in names:
    dataname = os.path.join(in_dir, name)
    feat = np.load(dataname)
    feat = feat[0, :, :]
    D, I = kmeans.index.search(feat, 1)
    I = np.array(I[:, 0])
    I = I.astype(np.float)
    faiss_k_means[name[:-4]] = I
    np.save(os.path.join(token_out_dir, name[:-4]+'.npy'), I)
kaldiio.save_ark('libri_token_faiss.ark', faiss_k_means, 'libri_token_faiss.scp', False)


