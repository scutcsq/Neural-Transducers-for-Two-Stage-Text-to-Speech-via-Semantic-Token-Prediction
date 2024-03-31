from typing import Callable, Dict, List, Sequence, Union

import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone

from collation import TextTokenCollater,AudioTokenCollater, MelCollater


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            text_token_collater: TextTokenCollater,
            audio_token_collater: AudioTokenCollater,
            mel_collater: MelCollater,
            cut_transforms: List[Callable[[CutSet], CutSet]] = None,
            # feature_input_strategy: BatchIO = PrecomputedFeatures(),
            # feature_transforms: Union[Sequence[Callable], Callable] = None,
    ) -> None:
        super().__init__()

        self.text_token_collater = text_token_collater
        self.audio_token_collater = audio_token_collater
        self.mel_collater = mel_collater
        self.cut_transforms = ifnone(cut_transforms, [])
        # self.feature_input_strategy = feature_input_strategy

        # if feature_transforms is None:
        #     feature_transforms = []
        # elif not isinstance(feature_transforms, Sequence):
        #     feature_transforms = [feature_transforms]

        # assert all(
        #     isinstance(transform, Callable) for transform in feature_transforms
        # ), "Feature transforms must be Callable"
        # self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)
        for transform in self.cut_transforms:
            cuts = transform(cuts)
        
        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom['tokens']['text'] for cut in cuts]
        )
        audio_tokens, audio_tokens_lens = self.audio_token_collater(
            [cut.supervisions[0].custom['wav2vec']['wav2vec'] for cut in cuts]
        )
        seg_mels = self.mel_collater(
            [cut.supervisions[0].custom['mel']['mel'] for cut in cuts]
        )
        return{
            "utt_id": [cut.id for cut in cuts],
            "text_seq": text_tokens,
            "text_len": text_tokens_lens,
            "token_seq": audio_tokens,
            "token_len": audio_tokens_lens,
            "ref_mels": seg_mels
        }






def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."