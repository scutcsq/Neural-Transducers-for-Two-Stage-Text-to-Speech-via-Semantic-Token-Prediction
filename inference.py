from itertools import count
import os
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from warnings import simplefilter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write
from stage1.stage1 import Stage1Net
from stage2.stage2 import SynthesizerTrn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from stage2.vits.utils import load_checkpoint
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from inference_dataset import TransducerDataset, TransducerCollate
from torch.nn.parallel import DistributedDataParallel as DDP
import json

MAX_WAV_VALUE = 32768.0
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
torch.cuda.empty_cache()
import soundfile as sf
SEED = 1


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train(rank, args, hparams_stage1, hparams_stage2):
    if hparams_stage1.num_gpus > 1:
        print('try to load multi gpu')
        os.environ['MASTER_ADDR'] = 'xxx.xxx.xxx.xxx'
        os.environ['MASTER_PORT'] = '52590'
        init_process_group(backend = hparams_stage1.dist_backend,
                           world_size=hparams_stage1.dist_world_size * hparams_stage1.num_gpus,
                           rank = rank)
        print('multi gpu loadding success!')
    device = torch.device('cuda:{:d}'.format(rank))

    stage1model = Stage1Net(text_dim = 384, 
                   num_vocabs = 513, 
                   num_phonemes = 512, 
                   token_dim = 256, 
                   hid_token_dim = 512, 
                   inner_dim = 513, 
                   ref_dim = 513, 
                   out_dim = 1024)
    pam = count_parameters(stage1model)
    print('stage1 pam: ', pam)

    stage2model = SynthesizerTrn(
        n_vocab = 513,
        spec_channels = 101,
        segment_size = 100,
    )
    pam2 = count_parameters(stage2model)
    print('stage2 pam: ', pam2)


    
    termination_symbol = 0

    iteration = 0
    valset = TransducerDataset(args.mel_scp, args.meta_file)
    collate_fn = TransducerCollate()
    

    #load model
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location = device)
    stage1model.load_state_dict(stage1_ckpt['state_dict'])
    stage2model, _ , _, _ = load_checkpoint(args.stage2_ckpt, stage2model)

    stage1model = stage1model.to(device)
    stage2model = stage2model.to(device)

    if hparams_stage1.num_gpus > 1:
        stage1model = DDP(stage1model, device_ids = [rank])
        stage2model = DDP(stage2model, device_ids = [rank])
    
    stage1model.eval()
    stage2model.eval()
    val_loader = DataLoader(valset,
                            sampler = None,
                            num_workers = 4,
                            shuffle = False,
                            batch_size = 1,
                            pin_memory = False,
                            collate_fn = collate_fn)
    pbar = tqdm(val_loader)
    for i, batch in enumerate(pbar):
        name, phone_padded, phone_seq_len, mels, tokens = batch
        
        phone_padded = phone_padded.to(device).long()
        phone_seq_len = phone_seq_len.to(device).long()
        mels = mels.to(device).float()
        tokens = tokens.to(device).long()

        with torch.no_grad():
            if hparams_stage1.num_gpus > 1:
                token_seq = stage1model.module.recognize(inputs = phone_padded, input_lens = phone_seq_len, reference_audio = mels)
                token_len = torch.zeros(1)
                token_len[:] = len(token_seq) * 2
                
                token_len = token_len.to(device).long()
                predict_audio, mask = stage2model.module.infer(token_seq, token_len, mels)
            else:
                token_seq = stage1model.recognize(inputs = phone_padded, input_lens = phone_seq_len, reference_audio = mels)
                token_len = torch.zeros(1)
                token_len[:] = token_seq.shape[1] * 2
                token_len = token_len.to(device).long()
                token_seq = token_seq.to(device).long()
                predict_audio, mask = stage2model.infer(token_seq, token_len, mels)
            ori_token_len = torch.zeros(1)
            ori_token_len[:] = tokens.shape[1] * 2
            ori_token_len = ori_token_len.to(device).long()
            
            with torch.no_grad():
                ori_audio, _ = stage2model.infer(tokens, ori_token_len, mels)
            ori_audio = ori_audio.cpu().data.numpy()
            wav = predict_audio.cpu().data.numpy()

            ori_output_file = os.path.join(args.result_dir, name[0] + '.wav')
            output_file = os.path.join(args.result_dir, name[0] + '_hat.wav')
            write(output_file, hparams_stage2.sampling_rate, wav)
            write(ori_output_file, hparams_stage2.sampling_rate, ori_audio)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_ckpt', type = str, default = 'stage1model_5e_4/ckpt/checkpoint_7343')
    parser.add_argument('--stage2_ckpt', type = str, default = 'stage2model/G_120000.pth')
    parser.add_argument('--result_dir', type = str, default = 'infer_dirs/')
    parser.add_argument('--mel_scp', type= str, default = 'mel16k.scp')
    parser.add_argument('--meta_file', type = str, default = 'infer.txt')
    parser.add_argument('--stage1_hparams', type = str, default = 'stage1/vctk_config.json')
    parser.add_argument('--stage2_hparams', type = str, default = 'stage2/vctk_config.json')
    

    args = parser.parse_args()

    with open(args.stage1_hparams, 'r') as f:
        stage1_data = f.read()
    stage1_config = json.loads(stage1_data)
    hparams_stage1 = Hparams(stage1_config)

    with open(args.stage2_hparams, 'r') as f:
        stage2_data = f.read()
    stage2_config = json.loads(stage2_data)
    hparams_stage2 = Hparams(stage2_config)


    torch.backends.cudnn.enabled = hparams_stage1.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams_stage1.cudnn_benchmark

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        hparams_stage1.num_gpus = torch.cuda.device_count()
        hparams_stage2.batch_size = int(hparams_stage1.batch_size / hparams_stage1.num_gpus)
    else:
        pass

    os.makedirs(args.result_dir, exist_ok=True)

    if hparams_stage1.num_gpus > 1:
        mp.spawn(train, nprocs = hparams_stage1.num_gpus, args = (args, hparams_stage1, hparams_stage2))
    
    else:
        train(0, args, hparams_stage1, hparams_stage2)

