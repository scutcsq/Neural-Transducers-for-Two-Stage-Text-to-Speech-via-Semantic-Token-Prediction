import torch
import torch.nn as nn
import numpy as np
## Text Encoder
from conformer import Conformer
from ref_enc import ECAPA_TDNN as ReferenceEncoder
import torch.nn.functional as F
from torch import Tensor
class TextEncoder(nn.Module):
    def __init__(self, 
                 in_dim : 384, 
                 n_text : 512):
        super().__init__()
        self.conformer = Conformer( 
                  encoder_dim = in_dim, 
                  conv_kernel_size = 5,
                  num_encoder_layers = 6)
        
        self.embedding = nn.Embedding(n_text,
                                        in_dim)
        self.inner_size = 513
        self.fc = nn.Linear(in_dim, self.inner_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.conformer(x)
        x = self.fc(x)
        return x
    

## Prediction Network
class PredictionEncoder(nn.Module):
    def __init__(self, 
                 in_dim = 256, 
                 hid_dim = 512, 
                 n_token = 512):
        super().__init__()
        self.lstm = nn.LSTM(batch_first = True, input_size = in_dim, hidden_size = hid_dim, num_layers = 2, bidirectional = False)
        self.embedding = nn.Embedding(n_token, in_dim)
        self.inner_size = 513
        self.fc = nn.Linear(hid_dim, self.inner_size)
    def forward(self, x):
        x = self.embedding(x)
        x, y  = self.lstm(x)
        x = self.fc(x)
        return x, y
    def inference(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden



## Reference Network
# self.ref_p = ReferenceEncoder(hid_C = hidden_channels, out_C = hidden_channels)


## Joint Net
class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)
class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style
        
        style = self.style(style_code)
        if len(input.size()) == 4:
            style = style.unsqueeze(1).unsqueeze(1)
        elif len(input.size()) == 3:
            style = style.unsqueeze(1)

        gamma, beta = style.chunk(2, dim=-1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class JointStyleBlock(nn.Module):
    def __init__(self,
                 ref_size: int = 512,
                 audio_size: int = 512,
                 ):
        super().__init__()
        self.fc = nn.Linear(ref_size, ref_size)
        self.ln = StyleAdaptiveLayerNorm(ref_size, audio_size)
    def forward(self, x, ref_audio):
        x1 = self.ln(x, ref_audio)
        x1 = self.fc(x1)
        return x + x1

class JointStyleNet(nn.Module):
    def __init__(self, 
                 ref_size: int = 512,
                 audio_size: int = 512,
                 num_layers: int = 3):
        super().__init__()
        
        self.layers = nn.ModuleList([JointStyleBlock(int(ref_size * 2), audio_size)
                                     for _ in range(num_layers)])
    def forward(self, x, ref_audio):
        for layer in self.layers:
            x = layer(x, ref_audio)
        return x
    
class JointNet(nn.Module):
    def __init__(self,
                 num_vocabs:int,
                 output_size:int = 1024,
                 inner_size:int = 512,
                 text_size:int = 512,
                 label_size:int = 512,
                 refer_size:int = 512):
        super().__init__()
        self.fc1 = nn.Linear(text_size, inner_size)
        self.fc2 = nn.Linear(label_size, inner_size)
        self.fc3 = nn.Linear(int(inner_size * 2), output_size)
        self.fc4 = nn.Linear(output_size, num_vocabs)
        self.tanh = nn.Tanh()
        self.jointstylenet = JointStyleNet(inner_size, refer_size)
        self.bos_token = [0]
        self.eos_token = [-1]
        self.pad_token = [-2]
        # reference_emb = self.refencoder(reference_audio)
        self.refencoder = ReferenceEncoder(hid_C = int(refer_size * 2), out_C = inner_size)
    def forward(self, text, label, reference):
        
        # text = self.fc1(text)
        # label = self.fc2(label)
        reference = self.refencoder(reference)
        if text.dim() == 3 and label.dim() == 3:
            seq_lens = text.size(1)
            tar_lens = label.size(1)
            
            text = text.unsqueeze(2)
            label = label.unsqueeze(1)

            text = text.repeat(1, 1, tar_lens, 1)
            label = label.repeat(1, seq_lens, 1, 1)
        
        hidden = torch.cat((text, label), dim = -1)
        hidden = self.jointstylenet(hidden, reference)
        
        out = self.fc3(hidden)
        out = self.tanh(out)
        out = self.fc4(out)
        out = F.log_softmax(out, dim = -1)
        return out

class Stage1Net(nn.Module):
    def __init__(self,
                 text_dim,
                 num_vocabs,
                 num_phonemes,
                 token_dim,
                 hid_token_dim,
                 inner_dim,
                 ref_dim,
                 out_dim):
        super().__init__()
        self.textencoder = TextEncoder(text_dim,
                                       num_phonemes)
        self.tokenencoder = PredictionEncoder(token_dim,
                                              hid_token_dim,
                                              num_vocabs,
                                              )
        self.refencoder = ReferenceEncoder(hid_C = ref_dim, out_C = inner_dim)
        self.JointNet = JointNet(num_vocabs,
                                 out_dim,
                                 inner_dim,
                                 text_dim,
                                 hid_token_dim,
                                 ref_dim,
                                 )
    def forward(self,
                text_seq: Tensor,
                token_seq: Tensor,
                reference_audio: Tensor
                ):
        text_seq = self.textencoder(text_seq)
        token_seq, _  = self.tokenencoder(token_seq)
        out = self.JointNet(text_seq, token_seq, reference_audio)
        return out, text_seq, token_seq
    
    @torch.no_grad()
    def decode(self, 
               text_outputs: Tensor,
               max_lens: int,
               reference_emb: Tensor):
        batch = text_outputs.size(0)
        y_hats = list()
        targets = torch.LongTensor([0] * batch).to(text_outputs.device)
        targets = targets.unsqueeze(-1)
        time_num = 0
        for i in range(int(max_lens)):
            pred = -1
            text_output = text_outputs[:, i, :].unsqueeze(1)
            while(pred != 0):
                if time_num == 0:
                    label_output, hidden = self.tokenencoder.inference(targets)
                else:
                    label_output, hidden = self.tokenencoder.inference(targets, hidden)
                
                output = self.JointNet(text_output, label_output, reference_emb)
                output = output.squeeze(1).squeeze(1)
                
                top_k_output_values, top_k_output_indices = torch.topk(output, k = 5, dim = -1)
                sum_values = torch.sum(top_k_output_values, dim = -1)
                
                normed_top_k_output_values = top_k_output_values / sum_values.unsqueeze(-1)
                choosed_indices = torch.multinomial(normed_top_k_output_values, num_samples=1)
                
                targets = top_k_output_indices[0, choosed_indices]
                
                pred = targets
                time_num += 1
                if pred == 0:
                    break
                else:
                    y_hats.append(targets[0,:])
        y_hats = torch.stack(y_hats, dim = 1)
        return y_hats
                 
    @torch.no_grad()
    def recognize(self, inputs, input_lens, reference_audio):
        text_outputs = self.textencoder(inputs)
        max_lens, _  = torch.max(input_lens, dim = -1)
        return self.decode(text_outputs, max_lens, reference_audio)
