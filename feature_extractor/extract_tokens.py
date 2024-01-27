import joblib
import os
import numpy as np
import kaldiio
km_path = r'/148Dataset/data-chen.shuaiqi/Transducer/km.bin'
kmeans_model = joblib.load(open(km_path, 'rb'))
# centers = kmeans_model.cluster_centers

in_dir = r'./VCTKdata/wav2vec2_15th/'
out_dir = r'./VCTKdata/token_wav2vec2_15th/'

os.makedirs(out_dir, exist_ok = True)
in_files = os.listdir(in_dir)
token_seq = {}
for data in in_files:
    file = os.path.join(in_dir, data)
    filename = data[:-4]
    tokens = np.load(file)
    print('tokens: ', tokens.shape)
    quan_tokens = kmeans_model.predict(tokens[0,:,:])
    quan_tokens = np.array(quan_tokens)
    out_file = os.path.join(out_dir, data)
    np.save(out_file, quan_tokens)
    token_seq[filename] = quan_tokens

kaldiio.save_ark('token.ark', token_seq, 'token.scp', False)
