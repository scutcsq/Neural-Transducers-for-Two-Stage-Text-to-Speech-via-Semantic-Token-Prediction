from gc import enable
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
# import IPython.display as ipd
import vits.commons as commons
import vits.utils as utils
from dataset import DistributedBucketSampler, InferenceCollate, InferenceDataset
from stage2 import SynthesizerTrn,  MultiPeriodDiscriminator
from vits.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import numpy as np
from torch.distributed import init_process_group

from vits.utils import load_checkpoint
from scipy.io.wavfile import write
import json

SEED = 1
torch.backends.cudnn.benchmark = True
global_step = 0

class Hparams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Hparams(**v)
            self[k] = v
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.values()
    def values(self):
        return self.__dict__.values()
    def __len__(self):
        return len(self.__dict__)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    def __contains__(self, key):
        return key in self.__dict__
    def __repr__(self):
        return self.__dict__.__repr__()

def train(rank, hparams, args):
    
    
    device = torch.device('cuda:{:d}'.format(rank))

    eval_dataset = TokenVocoderDataset(hparams, args.meta_file, args.token_scp, args.mel_scp, 'valid')
    collate_fn = TokenVocoderCollate()
    eval_loader = DataLoader(eval_dataset, num_workers = 8, shuffle = False, batch_size = 1, pin_memory = True, drop_last = False, collate_fn = collate_fn)
    net_g = SynthesizerTrn(
        n_vocab = 513,
        spec_channels = 101,
        segment_size = 100,
    )

    net_g, _, _, _ = load_checkpoint(args.cp_path, net_g)

    net_g = net_g.to(device)

    net_g.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            token_padded, token_lens, mels, audio_padded, audio_lens, _, _, name = batch

            token_padded = token_padded.to(device).long() 
            token_lens = token_lens.to(device).long()
            mels = mels.to(device).float()
            audio_padded = audio_padded.to(device).float()
            audio_lens = audio_lens.to(device).long()

            audio_padded = audio_padded.unsqueeze(0)
            # print('token_padded: ', token_padded.shape, token_padded, 'lens: ', token_lens.shape, token_lens)
            y_hat, mask = net_g.infer(token_padded, token_lens, mels)

            # y_hat = y_hat.squeeze(1)

            y_hat = y_hat.cpu().data.numpy()
            audio_padded = audio_padded.cpu().data.numpy()

            ori_file = os.path.join(args.result_dir, name[0] + '.wav')
            hat_file = os.path.join(args.result_dir, name[0] + '_hat.wav')
            print('audio: ', audio_padded.shape, 'y_hat: ', y_hat.shape)
            write(ori_file, hparams.sampling_rate, audio_padded)
            write(hat_file, hparams.sampling_rate, y_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cp_path', type = str, default = 'stage2model/G_120000.pth')
    parser.add_argument('--result_dir', type = str, default = 'stage2_infer_dirs/')
    parser.add_argument('--mel_scp', type= str, default = 'mel16k.scp')
    parser.add_argument('--meta_file', type = str, default = 'infer.txt')
    parser.add_argument('--token_scp', type = str, default ='token.scp')
    parser.add_argument('--config_path', type = str, default = 'vctk_config.json')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        hparams.num_gpus = torch.cuda.device_count()
        hparams.batch_size = int(hparams.batch_size / hparams.num_gpus)
    else:
        pass

    os.makedirs(args.result_dir, exist_ok=True)

    if hparams.num_gpus > 1:
        mp.spawn(train, nprocs = hparams.num_gpus, args = (hparams, args))
    
    else:
        train(0, hparams, args)
