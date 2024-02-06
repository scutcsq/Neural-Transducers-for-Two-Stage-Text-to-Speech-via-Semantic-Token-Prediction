from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import os
import soundfile
import numpy as np
import joblib
#----------------------------------------
# extract wav2vec features
ckpt_path = r'./wav2vec2-xlsr-53-espeak-cv-ft/'
device = torch.device('cuda:3')
processor = Wav2Vec2Processor.from_pretrained(ckpt_path)
model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
model = model.to(device)
in_dir = r'VCTKdata/totalwav/'
out_dir = r'VCTKdata/wav2vec2_15th/'
wavs = os.listdir(in_dir)
layer = 15
model.eval()
for wav in wavs:
    if wav[-8:] == 'mic1.wav':
        audio_input, sr = soundfile.read(os.path.join(in_dir, wav))
        input_values = processor(audio_input, sampling_rate = sr, return_tensors = 'pt').input_values
        input_values = input_values.to(device)
        with torch.no_grad():
            output = model(input_values, output_hidden_states = True)
        result = output.hidden_states[layer] #(1, T, 1024)
        result = result.cpu().data.numpy()
        save_name = os.path.join(out_dir, wav[:-4] + '.npy')
        np.save(save_name, result)
