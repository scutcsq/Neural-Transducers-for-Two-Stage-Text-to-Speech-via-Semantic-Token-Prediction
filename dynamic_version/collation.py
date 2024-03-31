from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# from valle.utils import SymbolTable
from simble_table import SymbolTable
import random

class TextTokenCollater:
    """Collate list of text tokens

    Map sentences to integers. Sentences are padded to equal length.
    Beginning and end-of-sequence symbols can be added.

    Example:
        >>> token_collater = TextTokenCollater(text_tokens)
        >>> tokens_batch, tokens_lens = token_collater(text)

    Returns:
        tokens_batch: IntTensor of shape (B, L)
            B: batch dimension, number of input sentences
            L: length of the longest sentence
        tokens_lens: IntTensor of shape (B,)
            Length of each sentence after adding <eos> and <bos>
            but before padding.
    """

    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
    ):
        self.pad_symbol = pad_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        unique_tokens = (
            [pad_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(text_tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = [token for token in unique_tokens]

    def index(
        self, tokens_list: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print('index:', tokens_list[0])
        seqs, seq_lens = [], []
        for tokens in tokens_list:
            assert (
                all([True if s in self.token2idx else False for s in tokens])
                is True
            )
            seq = (
                ([self.bos_symbol] if self.add_bos else [])
                + list(tokens)
                + ([self.eos_symbol] if self.add_eos else [])
            )
            seqs.append(seq)
            seq_lens.append(len(seq))

        max_len = max(seq_lens)
        for k, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
            seq.extend([self.pad_symbol] * (max_len - seq_len))

        tokens = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )
        tokens_lens = torch.IntTensor(seq_lens)

        return tokens, tokens_lens

    def __call__(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        print('call: ', texts[0])
        tokens_seqs = [[p for p in text] for text in texts]
        max_len = len(max(tokens_seqs, key=len))

        seqs = [
            ([self.bos_symbol] if self.add_bos else [])
            + list(seq)
            + ([self.eos_symbol] if self.add_eos else [])
            + [self.pad_symbol] * (max_len - len(seq))
            for seq in tokens_seqs
        ]

        tokens_batch = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq) + int(self.add_eos) + int(self.add_bos)
                for seq in tokens_seqs
            ]
        )

        return tokens_batch, tokens_lens


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater


class AudioTokenCollater:
    def __init__(
            self,
            pad_id: int = 0
    ):
        self.pad_id = pad_id

    def index(
            self, tokens_list: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs, seq_lens = [], []
        for tokens in tokens_list:
            seq = (
                ([self.pad_id])
                + list(tokens)
            )
            seqs.append(seq)
            seq_lens.append(len(seqs) - 1)
        
        max_len = max(seq_lens)
        for k, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
            seq.extend([self.pad_id] * (max_len - seq_len))

        tokens = torch.from_numpy(
            np.array([[seq] for seq in seqs]),
            dtype = np.int64
        )
        tokens_lens = torch.IntTensor(seq_lens)

        return tokens, tokens_lens
    
    def __call__(self, audios: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_seqs = [[p for p in audio] for audio in audios]

        max_len = len(max(tokens_seqs, key = len))

        seqs = [
            ([self.pad_id])
            + list(seq)
            + [self.pad_id] * (max_len - len(seq))
            for seq in tokens_seqs
        ]

        tokens_batch = torch.from_numpy(
            np.array(
                [[seq] for seq in seqs],
                dtype = np.int64
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq) 
                for seq in tokens_seqs
            ]
        )
        return tokens_batch, tokens_lens
    

class MelCollater:
    def __init__(
            self,
            segment_size: int=281   
    ):
        self.segment_size = segment_size
    def index(
            self,
            mels_list: List[float]
    ) -> torch.Tensor:
        mels = []
        for mel in mels_list:
            mel = torch.numpy(mel)
            mel_start = random.randint(0, mel.size(0))
            seg_mel = mel[mel_start: mel_start + self.segment_size, :]

            if seg_mel.size(0) < self.segment_size:
                seg_mel = seg_mel.permute([1, 0])
                seg_mel = torch.nn.functional.pad(seg_mel, (0, self.segment_size - seg_mel.size(1)), 'constant')
            
            else:
                seg_mel = seg_mel.permute([1, 0])
            seg_mel = seg_mel.data.numpy()
            mels.append(seg_mel)
        mels = torch.from_numpy(
            np.array(mels)
        )
        return mels
    
    def __call__(self,
                 mels_list: List[float]):
        
        mel_list = [[mel] for mel in mels_list]
        mels = []
        for mel in mel_list:
            mel = torch.Tensor(mel)
            mel_start = random.randint(0, mel.size(0))
            seg_mel = mel[0, mel_start: mel_start + self.segment_size, :]
            if seg_mel.size(0) < self.segment_size:
                seg_mel = seg_mel.permute([1, 0])
                seg_mel = torch.nn.functional.pad(seg_mel, (0, self.segment_size - seg_mel.size(1)), 'constant')
            
            else:
                seg_mel = seg_mel.permute([1, 0])
            
            seg_mel = seg_mel.data.numpy()
            mels.append(seg_mel)
        
        mels = torch.from_numpy(
            np.array(mels)
        )
        return mels
    


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater



def get_audio_token_collater() -> AudioTokenCollater:
    collater = AudioTokenCollater()
    return collater

def get_mel_collater():
    collater = MelCollater()
    return collater