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

from stage1 import Stage1Net
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import TransducerDataset, TransducerCollate
import kaldiio
import k2
import json
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
torch.cuda.empty_cache()
# from warprnnt_pytorch import RNNTLoss

from scipy.io import wavfile
from scipy import signal
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

def save_checkpoint(model, optimizer, lr, iter, filepath):
    torch.save(
        {
            'iteration': iter,
            'state_dict': model.state_dict(),
            'lr': lr,
            'optimizer': optimizer.state_dict()
        },
        filepath
    ) 
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
def train(rank, args, hparams):
    if args.num_gpus > 1:
        print('try to load multi gpu')
        os.environ['MASTER_ADDR'] = 'xxx.xxx.xxx.xxx'
        os.environ['MASTER_PORT'] = '52590'
        init_process_group(backend = hparams.dist_backend,
                           world_size=hparams.dist_world_size * args.num_gpus,
                           rank = rank)
        print('multi gpu loadding success!')
    device = torch.device('cuda:{:d}'.format(rank))
    # device = torch.device('cpu')

    model = Stage1Net(text_dim = 384, 
                   num_vocabs = 513, 
                   num_phonemes = 512, 
                   token_dim = 256, 
                   hid_token_dim = 512, 
                   inner_dim = 513, 
                   ref_dim = 513, 
                   out_dim = 1024)
    pam = count_parameters(model)
    print('pam: ', pam)
    model = model.to(device)
    # out_size = fix_len_com
    
    # criterion = RNNTLoss()
    termination_symbol = 0
    
    # Dataloader:
    trainset = TransducerDataset(args.mel_scp, args.token_scp, args.train_meta, 'train')
    valset = TransducerDataset(args.mel_scp, args.token_scp, args.valid_meta, 'valid')
    collate_fn = TransducerCollate()
    
    if args.num_gpus > 1:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(trainset,
                              num_workers = 4,
                              shuffle = shuffle,
                              sampler = train_sampler,
                              batch_size = hparams.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=collate_fn)
    
    iteration = 0
    accum_iteration = 0
    epoch_offset = 0
    initial_lr = hparams.initial_lr
    if args.checkpoint_path is not None:
        assert os.path.isfile(args.checkpoint_path), f"{args.checkpoint_path} is not existed"
        checkpoint_dict = torch.load(args.checkpint_path, map_location = device)
        
        model.load_state_dict(checkpoint_dict['state_dict'])
        print('loading existing model successfully')
        if args.load_iteration:
            initial_lr = checkpoint_dict['lr']
            iteration = checkpoint_dict['iteration']
            iteration += 1
            epoch_offset = int(iteration / len(train_loader))
        
        print('loading successful')
    
    if args.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=True).to(device)
        
    #optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr = initial_lr,
                                  betas=[0.9, 0.98],
                                  amsgrad = True)
    
    if args.checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    for params in optimizer.param_groups:
        params['initial_lr'] = initial_lr
    
    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer,
    #     gamma = hparams.lr_decay,
    #     last_epoch=epoch_offset
    # )    
    
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer, T_0 = 10, T_mult = 2, eta_min = 0)
    model.train()
    accumulation_step = 64
    count_step = 1
    summary_interval = 500
    optimizer.zero_grad()
    sw = SummaryWriter(os.path.join(args.output_directory, 'logs'))
    for epoch in range(epoch_offset, hparams.epoches):
        print('Epoch: {}, learning rate: {}'.format(epoch, scheduler_g.get_last_lr()))
        
        if args.num_gpus >1:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        batch_num = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            start = time.perf_counter()
            
            # phone_padded, duration_padded, text_lengths, \
            #     mel_padded, mel_lengths, speaker_ids, emotion_ids, gender_ids, pitch_ids, tempo_ids, energy_ids, style_padded, rob_padded, rob_mask, word2phone_padded, _  = batch
            phone_padded, token_padded, phone_seq_len, token_seq_len, mels = batch
            
            phone_padded = phone_padded.to(device).long()
            token_padded = token_padded.to(device).long()
            phone_seq_len = phone_seq_len.to(device).long()
            token_seq_len = token_seq_len.to(device).long()
            
            mels = mels.to(device).float()

            
            boundary = torch.zeros((phone_padded.size(0), 4), dtype = torch.int64).to(device)
            # print('token_padded: ', token_padded.shape[1], 'phone_padded: ', phone_padded.shape[1])
            boundary[:, 2] = token_seq_len
            boundary[:, 3] = phone_seq_len
            
            
            
            result, texts, tokens = model(phone_padded, token_padded, mels)
            
            # print('tokens:', tokens.shape)
            # print('text: ', texts.shape)
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm = tokens,
                am = texts,
                symbols = token_padded[:, 1:],
                # symbols = token_padded,
                termination_symbol = termination_symbol,
                lm_only_scale = 0.25,
                am_only_scale = 0.0,
                boundary = boundary,
                reduction = 'none',
                return_grad = True
            )
            s_range = 50

            ranges = k2.get_rnnt_prune_ranges(
                px_grad = px_grad,
                py_grad = py_grad,
                boundary = boundary,
                s_range = s_range
            )
            texts_pruned, token_pruned = k2.do_rnnt_pruning(am = texts, lm = tokens, ranges = ranges)
            
            if args.num_gpus > 1:
                logits = model.module.JointNet(texts_pruned, token_pruned, mels)
            else:
                logits = model.JointNet(texts_pruned, token_pruned, mels)
            # print('logits: ', logits.shape)
            pruned_loss = k2.rnnt_loss_pruned(
                logits = logits,
                symbols = token_padded[:, 1:],
                # symbols = token_padded,
                boundary = boundary,
                ranges = ranges,
                termination_symbol = termination_symbol,
                reduction = 'none'
            )
            # pruned_loss.backward()
            simple_loss_is_finite = torch.isfinite(simple_loss)
            pruned_loss_is_finite = torch.isfinite(pruned_loss)
            is_finite = simple_loss_is_finite & pruned_loss_is_finite
            if not torch.all(is_finite):
                
                simple_loss = simple_loss[simple_loss_is_finite]
                pruned_loss = pruned_loss[pruned_loss_is_finite]

                # If either all simple_loss or pruned_loss is inf or nan,
                # we stop the training process by raising an exception
                if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite):
                    raise ValueError(
                        "There are too many utterances in this batch "
                        "leading to inf or nan losses."
                    )

            simple_loss = simple_loss.sum()
            pruned_loss = pruned_loss.sum()
            loss = pruned_loss + 0.5 * simple_loss
            show_loss = loss.clone()
            
            loss /= accumulation_step

            epoch_loss += loss

            loss.backward()
            
            if ((iteration+1) % accumulation_step) == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                batch_num += 1
                accum_iteration += 1
                if rank == 0:
                    timeduration = time.perf_counter() - start
                    pbar.set_description(
                        "Train {} total_loss{:.4f}".format(accum_iteration, show_loss.item()
                    ))

            if rank == 0 and (iteration % hparams.saved_checkpoint) == 0:
                print('save checkpint...')
                checkpoint_path = os.path.join(
                args.output_directory, "ckpt",
                "checkpoint_{}".format(accum_iteration))
                save_checkpoint(model, optimizer, optimizer.param_groups[0]['lr'], iteration, checkpoint_path)
            if iteration % summary_interval == 0:
                sw.add_scalar('rnn_loss', show_loss, iteration)
                

            iteration += 1
             
            if rank == 0 and iteration % hparams.evaluate_epoch == 0:
                print("Evaluate")
                model.eval()
                countnum = 0
                with torch.no_grad():
                    val_loader = DataLoader(valset,
                                            sampler = None,
                                            num_workers = 4,
                                            shuffle = False,
                                            batch_size = 1,
                                            pin_memory = False,
                                            collate_fn = collate_fn)
                    
                    pbar2 = tqdm(val_loader)
                    for i, batch in enumerate(pbar2):
                        
                        phone_padded, token_padded, phone_seq_len, token_seq_len, mels = batch
                        phone_padded = phone_padded.to(device).long()
                        token_padded = token_padded.to(device).long()
                        phone_seq_len = phone_seq_len.to(device).long()
                        token_seq_len = token_seq_len.to(device).long()
                        mels = mels.to(device).float()

                        if args.num_gpus > 1:
                            result = model.module.recognize(inputs = phone_padded, input_lens = phone_seq_len, reference_audio = mels)
                        else:
                            result = model.recognize(inputs = phone_padded, input_lens = phone_seq_len, reference_audio = mels)
                        
                        
                        countnum += 1
                        os.makedirs(os.path.join(args.output_directory, str(accum_iteration)), exist_ok= True)
                        np.save(os.path.join(args.output_directory, str(accum_iteration), str(i)+ '.npy'), result.cpu().data.numpy)
                        if countnum == 5:
                            model.train()
                            break
            
        epoch_loss /= batch_num
        
        print("Epoch {}: Train {} total_loss: {:.4f}".format(epoch, accum_iteration, epoch_loss.item()))
        scheduler_g.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint_path', type = str, default = None)
    parser.add_argument('--train_meta', type = str, default = r'total.txt')
    parser.add_argument('--valid_meta', type = str, default = r'total.txt')
    parser.add_argument('--mel_scp', type= str, default = r'mel16k.scp')
    parser.add_argument('--token_scp', type = str, default = r'token.scp')
    parser.add_argument('--load_iteration', type = bool, default = True)
    parser.add_argument('--output_directory', type = str, default = r'stage1model/')
    parser.add_argument('--config_path', type = str, default = r'vctk_config.json')
    
    args = parser.parse_args()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        args.num_gpus = torch.cuda.device_count()
        print('gpu_nums: ', args.num_gpus)
        args.batch_size = int(args.batch_size / args.num_gpus)
    else:
        pass
        
    os.makedirs(os.path.join(args.output_directory, 'ckpt'), exist_ok = True)
    if args.num_gpus > 1:
        print('run!')
        mp.spawn(train, nprocs=args.num_gpus, args = (args, hparams))
    else:
        train(0, args, hparams)
