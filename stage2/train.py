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
from dataset import DistributedBucketSampler, TokenVocoderCollate, TokenVocoderDataset
from stage2 import SynthesizerTrn,  MultiPeriodDiscriminator
from vits.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import numpy as np
from torch.distributed import init_process_group
# from text.symbols import symbols
from scipy.io.wavfile import write
from tqdm import tqdm
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

def train(rank, args, hparams):
    if rank == 0:
        logger = utils.get_logger(args.model_dir)
        utils.check_git_hash(args.model_dir)

        writer = SummaryWriter(log_dir = args.model_dir)
        writer_eval = SummaryWriter(log_dir = os.path.join(args.model_dir, 'eval'))
    if args.num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'xxx.xxx.xxx.xx'
        os.environ['MASTER_PORT'] = '55585'
        init_process_group(backend = hparams.dist_backend,
                           world_size = hparams.dist_world_size * args.num_gpus,
                           rank = rank)
    device = torch.device('cuda:{:d}'.format(rank))
    # device = torch.device('cpu')

    train_dataset = TokenVocoderDataset(args, 'train')
    train_sampler = DistributedBucketSampler(train_dataset, hparams.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas = hparams.num_gpus, rank = rank, shuffle = True)
    collate_fn = TokenVocoderCollate()

    train_loader = DataLoader(train_dataset, num_workers = 8, shuffle = False, pin_memory = True, collate_fn = collate_fn, batch_sampler = train_sampler)




    if rank == 0:
        eval_dataset = TokenVocoderDataset(args, 'valid')
        eval_loader = DataLoader(eval_dataset, num_workers = 8, shuffle = False, batch_size = 1, pin_memory = True, drop_last = False, collate_fn = collate_fn)

    net_g = SynthesizerTrn(
        n_vocab = 513,
        spec_channels = 101,
        segment_size = 100,
    ).to(device)
    net_d = MultiPeriodDiscriminator().to(device)
    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hparams.learning_rate,
                                betas = hparams.betas,
                                eps = hparams.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hparams.learning_rate,
        betas = hparams.betas,
        eps = hparams.eps
    )
    if args.num_gpus > 1:
        net_g = DDP(net_g, device_ids = [rank])
        net_d = DDP(net_d, device_ids = [rank])

    # try:
    #     _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(args.model_dir, 'G_*.pth'), net_g, optim_g)
    #     _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(args.model_dir, 'D_*.pth'), net_d, optim_d)
    # except:
    epoch_str = 1
    global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma = hparams.lr_decay, last_epoch = epoch_str - 2 )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma = hparams.lr_decay, last_epoch = epoch_str - 2 )

    scalar = GradScaler(enabled=hparams.fp16_run)
    for epoch in range(epoch_str, hparams.epochs + 1):
        if rank == 0:
            train_and_evaluate(device, rank, epoch, args, hparams, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scalar, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(device, rank, epoch, args, hparams, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scalar, [train_loader, None], None, None)
        scheduler_d.step()
        scheduler_g.step()

def train_and_evaluate(device, rank, epoch, args, hparams, nets, optims, schedulers, scalar, loaders, logger, writers):
    global global_step
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)

    pbar = tqdm(train_loader)
    net_g.train()
    net_d.train()
    # print('net_g: ', net_g.device())
    # print('net_d: ', net_d.device())
    for batch_idx, batch in enumerate(pbar):
        token_padded, token_lens, mels, audio_padded, audio_lens, spec_padded, spec_lens = batch
        token_padded = token_padded.to(device).long()
        token_lens = token_lens.to(device).long()
        mels = mels.to(device).float()
        audio_padded = audio_padded.to(device).float()
        audio_lens = audio_lens.to(device).long()
        spec_padded = spec_padded.to(device).float()
        spec_lens = spec_lens.to(device).long()

        with autocast(enabled = hparams.fp16_run):
            y_hat, token_length, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(token_padded, token_lens, spec_padded, spec_lens, mels)
            y_hat = y_hat.squeeze(1)
            mel = spec_to_mel_torch(
                spec_padded,
                hparams.filter_length,
                hparams.n_mel_channels,
                hparams.sampling_rate,
                hparams.mel_fmin,
                hparams.mel_fmax
            )
            
            y_mel = commons.slice_segments(mel, ids_slice, hparams.segment_size // hparams.hop_length)
            result_min_size = min(y_hat.size(1), audio_padded.size(1))
            
            y_hat = y_hat[:, :result_min_size]
            
            audio_padded = audio_padded.unsqueeze(1)
            y_hat_mel = mel_spectrogram_torch(
                y_hat,
                hparams.filter_length,
                hparams.n_mel_channels,
                hparams.sampling_rate,
                hparams.hop_length,
                hparams.win_length,
                hparams.mel_fmin,
                hparams.mel_fmax
            )
            y_hat = y_hat.unsqueeze(1)
            # print('ids_slice: ', ids_slice)
            # print('audio_padded: ', audio_padded.shape)
            # print('segment_size: ', hparams.segment_size)
            audio_padded = commons.slice_segments(audio_padded, ids_slice * hparams.hop_length, hparams.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(audio_padded, y_hat.detach())

            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scalar.scale(loss_disc_all).backward()
        scalar.unscale_(optim_d)

        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scalar.step(optim_d)


        with autocast(enabled=hparams.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(audio_padded, y_hat)
            with autocast(enabled = False):
                # print('y_mel: ', y_mel.shape, ' y_hat_mel: ', y_hat_mel.shape)
                # print('z_p: ', z_p.shape, ' logs_q: ', logs_q.shape, ' m_p: ', m_p.shape, ' logs_p: ', logs_p.shape, ' z_mask: ', z_mask.shape)
                # print('fmap_r: ', fmap_r.shape, ' fmap_g: ', fmap_g.shape)
                min_len = min(logs_p.shape[2], logs_q.shape[2])
                z_p = z_p[:, :, :min_len]
                logs_q = logs_q[:, :, :min_len]
                m_p = m_p[:, :, :min_len]
                logs_p = logs_p[:, :, :min_len]
                z_mask = z_mask[:, :, :min_len]

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hparams.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hparams.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scalar.scale(loss_gen_all).backward()
        scalar.unscale_(optim_g)

        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)

        scalar.step(optim_g)
        scalar.update()
        
        if rank == 0:
            if global_step % hparams.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('Train Epoch: {}[{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])
                scalar_dict = {'loss/g/total': loss_gen_all, 'loss/d/total':loss_disc_all, 'learning_rate':lr, 'grad_norm_d':grad_norm_d,'grad_norm_g':grad_norm_g}
                scalar_dict.update({'loss/g/fm':loss_fm, 'loss/g/mel':loss_mel, 'loss/g/kl':loss_kl, })
                scalar_dict.update({'loss/g/{}'.format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({'loss/d_r/{}'.format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({'loss/d_g/{}'.format(i): v for i, v in enumerate(losses_disc_g)})

                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                    scalars=scalar_dict
                )

                if global_step % hparams.eval_interval == 0:
                    evaluate(device, rank, args, hparams, net_g, eval_loader, writer_eval)
                    utils.save_checkpoint(net_g, optim_g, hparams.learning_rate, epoch, os.path.join(args.model_dir, 'G_{}.pth'.format(global_step)))
                    utils.save_checkpoint(net_d, optim_d, hparams.learning_rate, epoch, os.path.join(args.model_dir, 'D_{}.pth'.format(global_step)))
            
        global_step +=1
    if rank == 0:
        logger.info('======> Epoch: {}'.format(epoch))
        

def evaluate(device, rank, args, hparams, generator, eval_loader, writer_eval):
    global global_step
    nums = 0
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            token_padded, token_lens, mels, audio_padded, audio_lens, spec_padded, spec_lens = batch

            token_padded = token_padded.to(device).long()
            token_lens = token_lens.to(device).long()
            mels = mels.to(device).float()
            audio_padded = audio_padded.to(device).float()
            audio_lens = audio_lens.to(device).long()
            spec_padded = spec_padded.to(device).float()
            spec_lens = spec_lens.to(device).long()
            # print('lens: ', int(audio_lens[0]))
            if args.num_gpus > 1:
                y_hat,  mask = generator.module.infer(token_padded, token_lens, mels, max_len = int(audio_lens[0]))
            else:
                y_hat,  mask = generator.infer(token_padded, token_lens, mels, max_len = int(audio_lens[0]))

            # y_hat = y_hat.squeeze(1)
            y_hat = y_hat.cpu().data.numpy()
            os.makedirs(os.path.join(args.result_dir, str(global_step)), exist_ok= True)
            out_path = os.path.join(args.result_dir, str(global_step), str(batch_idx) + '.wav')
            
            write(out_path, hparams.sampling_rate, y_hat)
            # ipd.display(ipd.Audio(y_hat, rate = hparams.sampling_rate, normalize=False))
            # np.save(os.path.join(args.result_dir, str(global_step), str(batch_idx) + '.wav'), y_hat)
            nums += 1
            if nums == 20:
                break
    generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--meta_file', type = str, default = 'total.txt')
    parser.add_argument('--token_scp', type = str, default = 'token.scp')
    parser.add_argument('--mel_scp', type = str, default = 'mel16k.scp')
    parser.add_argument('--model_dir', type = str, default = 'stage2model')
    parser.add_argument('--result_dir', type = str, default = 'result_audio')
    parser.add_argument('--config_path', type = str, default = r'vctk_config.json')

    args = parser.parse_args()



    with open(args.config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        args.num_gpus = torch.cuda.device_count()
        print('gpu_nums: ', args.num_gpus)
    else:
        print('no gpu !!!')
        pass

    os.makedirs(args.model_dir, exist_ok= True)
    if args.num_gpus > 1:
        print('run')
        mp.spawn(train, nprocs = args.num_gpus, args = (args, hparams))
    else:
        train(0, args, hparams)
