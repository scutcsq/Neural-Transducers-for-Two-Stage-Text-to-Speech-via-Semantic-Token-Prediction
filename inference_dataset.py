import os
import sys


import random

import torch
import torch.utils.data
import tqdm
import kaldiio

from phone import symbols
import tqdm
import numpy as np




class TransducerDataset(torch.utils.data.Dataset):
    def __init__(self, mel_scp, meta_file):
        super().__init__()
        
        self.mel_dict = kaldiio.load_scp(mel_scp)
        self.token_dict = kaldiio.load_scp(r'token.scp')
        
        self.bos_token = torch.LongTensor([0])
        # self.eos_token = torch.LongTensor([513])
        # self.pad_token = torch.LongTensor([514])
        self.segment_size = 187
        
        self.meta_list = self.parse_meta_file(meta_file)
    def parse_meta_file(self, meta_file):
        meta_list = []
        self.phone_set = symbols
        phone_id_list = {s:i for i, s in enumerate(self.phone_set)}
        dir_name = ['p261', 'p225', 'p294', 'p347', 'p238', 'p234', 'p248', 'p335', 'p245', 'p326', 'p302']
        
        with open(meta_file, 'r', encoding = 'utf-8') as f_meta:
            meta_lines = f_meta.read().splitlines()
        for line in tqdm.tqdm(meta_lines):
            temp_list = []
            uttr_id, ref_uttr_id, dataset, spk, emo, gender, pitch, tempo, energy, feat = line.split('/', 9)
            
            transcirpts = []
            feat_list = feat.strip().split('\t')
            for unit in feat_list:
                phone, dur = unit.strip().split('|', 1)
                transcirpts.append(phone)
            # new_transcripts = ["<BOS>"]
            new_transcripts = []
            new_transcripts.extend(transcirpts)
            # new_transcripts.append("<EOS>")
            
            
            temp_list.append(uttr_id)
            temp_list.append(ref_uttr_id)
            temp_list.append([phone_id_list[phone] for phone in new_transcripts])
            meta_list.append(temp_list)
        return meta_list
    
    def __getitem__(self, index):
        uttr_id, ref_uttr_id, phone = self.meta_list[index]
        
        mels = torch.from_numpy(self.mel_dict[ref_uttr_id]) # (T, N)
        
        mels_start = random.randint(0, mels.size(0))
        
        phone_ids = torch.LongTensor(phone)
        
        segment_mels = mels[mels_start : mels_start + self.segment_size, : ]
        
        token = torch.from_numpy(self.token_dict[uttr_id]) + 1
        
        if segment_mels.size(0) < self.segment_size:
            segment_mels = segment_mels.permute([1, 0])
            segment_mels = torch.nn.functional.pad(segment_mels, (0, self.segment_size - segment_mels.size(1)), 'constant')
            # segment_mels = segment_mels.permute([1, 0])
            assert segment_mels.size(0) == 80
            assert segment_mels.size(1) == self.segment_size
        else:
            segment_mels = segment_mels.permute([1, 0])
            
        return (phone_ids, segment_mels, token, uttr_id)
    
    def __len__(self):
        return len(self.meta_list)
    
    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self), size=size, replace = False))
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch
    
class TransducerCollate():
    def __init__(self):
        super().__init__()
        
    def __call__(self, batch):
        input_lengths, idx_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim = 0,
            descending = True
        )
        
        max_input_len = input_lengths[0]
        max_token_len = max(x[2].size(0) for x in batch)
        phone_padded = torch.LongTensor(len(batch), max_input_len).zero_()
        
        mels = torch.FloatTensor(len(batch), 80, 187)
        
        phone_seq_len = torch.LongTensor(len(batch))
        
        uttr_id_list = []
        
        tokens = torch.LongTensor(len(batch), max_token_len)
        
        for i in range(len(idx_sorted_decreasing)):
            phone = batch[idx_sorted_decreasing[i]][0]
            phone_padded[i, :phone.size(0)] = phone
            phone_seq_len[i] = phone.size(0)
            token = batch[idx_sorted_decreasing[i]][2]
            tokens[i, :token.size(0)] = token
            mel = batch[idx_sorted_decreasing[i]][1]
            mels[i, :, :] = mel
            
            uttr_id_list.append(batch[idx_sorted_decreasing[i]][3])
        return (uttr_id_list, phone_padded, phone_seq_len, mels, tokens)