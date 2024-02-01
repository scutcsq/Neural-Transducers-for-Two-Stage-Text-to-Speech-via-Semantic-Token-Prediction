import os
import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import kaldiio
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr = sampling_rate, n_fft = n_fft, n_mels = num_mels, fmin = fmin, fmax = fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def extract_mel(in_path, out_path, dataset):
    os.makedirs(out_path, exist_ok=True)
    wavs = os.listdir(in_path)
    sampling_rate = 16000
    n_fft = 1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 8000
    for i in range(len(wavs)):
        if (os.path.exists(os.path.join(out_path, wavs[i][:-4] + '.npy'))):
            continue
        #VCTK:
        if dataset == 'vctk':
          if(wavs[i][-8:] == 'mic1.wav'):
              audio, sr = load_wav(os.path.join(in_path, wavs[i]))
              if(sr != 16000):
                  print('sr: ', sr)
              else:
                  audio = audio / MAX_WAV_VALUE
                  audio = normalize(audio) * 0.95
                  audio = torch.FloatTensor(audio)
                  audio = audio.unsqueeze(0)
                  mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
                  # mel = mel.squeeze(0)
                  mel = mel.permute([0,2,1])
                  mel = mel.squeeze(0)
                  mel = mel.data.numpy()
                  # print('mel: ', mel.shape)
                  np.save(os.path.join(out_path, wavs[i][:-4] + '.npy'), mel)
        #Libritts:
        elif dataset == 'libritts':
            if(wavs[i][-4:] == '.wav'):
              audio, sr = load_wav(os.path.join(in_path, wavs[i]))
              if(sr != 16000):
                  print('sr: ', sr)
              else:
                  audio = audio / MAX_WAV_VALUE
                  audio = normalize(audio) * 0.95
                  audio = torch.FloatTensor(audio)
                  audio = audio.unsqueeze(0)
                  mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
                  # mel = mel.squeeze(0)
                  mel = mel.permute([0,2,1])
                  mel = mel.squeeze(0)
                  mel = mel.data.numpy()
                  # print('mel: ', mel.shape)
                  np.save(os.path.join(out_path, wavs[i][:-4] + '.npy'), mel)
def mel2ark(in_path):
  dict = {}
  filelist = os.listdir(in_path)
  for wav in filelist:
    mel = np.load(os.path.join(in_path, wav))
    dict[wav[:-4]] = mel
  kaldiio.save_ark('mel16k.ark', scps, 'mel16k.scp', False)
    
in_path = r''
out_path = r''
extrace_mel(in_path, out_path, 'vctk')
mel2ark(out_path)
