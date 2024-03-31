#整个模型由4个部分组成, Text Encoder, Prediction Network, Reference Network, Joint Net

from tkinter import Scale
import torch
import torch.nn as nn
import numpy as np
## Text Encoder
from conformer import Conformer
from ref_enc import ECAPA_TDNN as ReferenceEncoder
import torch.nn.functional as F
from torch import Tensor
import ice_conformer.conformer as conformer
from ice_conformer.scaling import ScaledLinear, ScaledEmbedding, ScaledConv1d
import k2
class TextEncoder(nn.Module):
    def __init__(self, 
                 in_dim : 384, 
                 n_text : 512,
                 layer_nums:int = 6):
        super().__init__()
        # self.conformer = Conformer( 
        #           encoder_dim = in_dim, 
        #           conv_kernel_size = 5,
        #           num_encoder_layers = 6)
        self.conformer = conformer.Conformer(num_features = in_dim,
                                   d_model = in_dim,
                                   cnn_module_kernel = 5,
                                   dim_feedforward = int(in_dim * 4),
                                   num_encoder_layers = layer_nums)

        self.embedding = nn.Embedding(n_text,
                                        in_dim)
        
        self.relu = nn.ReLU()
    def forward(self, x, x_lens, warmup = 1.0):
        x = self.embedding(x)
        layer_results, x_lens = self.conformer(x, x_lens)
        encoder_out = layer_results[-1]
        # encoder_out = F.relu(encoder_out)
        # print('encoder: ', encoder_out.shape)
        return encoder_out
    

## Prediction Network
class PredictionEncoder(nn.Module):
    def __init__(self, 
                 in_dim = 256, 
                 hid_dim = 512, 
                 n_token = 512):
        super().__init__()
        self.lstm = nn.LSTM(batch_first = True, input_size = in_dim, hidden_size = hid_dim, num_layers = 2, bidirectional = False)
        self.embedding = nn.Embedding(n_token, in_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.num_layers = 2
        self.inner_size = 513
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(hid_dim, self.inner_size)
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        x, y  = self.lstm(x, (h0, c0))
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc(x)
        return x, y
    def inference(self, x, hidden=None):
        x = self.embedding(x)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
            x, hidden = self.lstm(x, (h0, c0))
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc(x)
        return x, hidden


class CNNPredictionNetwork(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 decoder_dim: int,
                 blank_id: int,
                 context_size:int):
        super().__init__()
        self.embedding = ScaledEmbedding(
            num_embeddings = vocab_size,
            embedding_dim = decoder_dim
        )
        self.blank_id = blank_id
        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(
                in_channels = decoder_dim,
                out_channels = decoder_dim,
                kernel_size = context_size,
                padding = 0,
                groups = decoder_dim,
                bias = False

            )
        else:
            self.conv = nn.Indentity()

    def forward(self,
                y: torch.Tensor
                ):
        if torch.jit.is_tracing():
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >=0).unsqueeze(-1)
        
        if self.context_size > 1:
            embedding_out = embedding_out.permute([0, 2, 1])
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out



## Reference Network
# self.ref_p = ReferenceEncoder(hid_C = hidden_channels, out_C = hidden_channels)


## Joint Net
    
class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                normal_shape = 513,
                epsilon=1e-5
                ):
        super().__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = normal_shape
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        # print('mean: ', mean.shape, '|var: ', var.shape, '|std: ', std.shape, '|y: ', y.shape)
        # print('speaker_embedding: ', speaker_embedding.shape)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        # print('scale: ', scale.shape, '| bias: ', bias.shape)
        y *= scale.unsqueeze(1).unsqueeze(1)
        y += bias.unsqueeze(1).unsqueeze(1)

        return y

class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        # affine = nn.Linear(in_dim, out_dim)
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
        self.fc = nn.Linear(ref_size, int(ref_size*4))
        # self.ln = StyleAdaptiveLayerNorm(ref_size, audio_size)
        self.ln2 = ConditionalLayerNorm(normal_shape=ref_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(int(ref_size * 4), ref_size)
    def forward(self, x, ref_audio):
        # x1 = self.ln(x, ref_audio)
        x1 = self.ln2(x, ref_audio)
        x1 = self.fc(x1)
        x1 = self.gelu(x1)
        x1 = self.fc2(x1)
        return x + x1

class JointStyleNet(nn.Module):
    def __init__(self, 
                 ref_size: int = 512,
                 audio_size: int = 512,
                 num_layers: int = 3):
        super().__init__()
        # self.layers = nn.ModuleList([nn.Sequential(StyleAdaptiveLayerNorm(ref_size, audio_size),
        #                                            nn.Linear(ref_size, ref_size))
        #                                            for _ in range(num_layers)])
        # self.fc = nn.Linear()
        self.layers = nn.ModuleList([JointStyleBlock(ref_size, audio_size)
                                     for _ in range(num_layers)])
    def forward(self, x, ref_audio):
        for layer in self.layers:
            x = layer(x, ref_audio)
        return x
    


class JointNet(nn.Module):
    def __init__(self,
                 encoder_dim:int = 384,
                 decoder_dim:int = 512,
                 reference_dim:int = 513,
                 vocab_size: int = 513
                 ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, 1024)
        self.decoder_proj = nn.Linear(decoder_dim, 1024)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.hidden_liner = ScaledLinear(vocab_size, hidden_dim)
        # self.output_linear = ScaledLinear(hidden_dim, vocab_size)
        self.output_linear = nn.Linear(1024, vocab_size)
        # self.jointstylenet = JointStyleNet(vocab_size, reference_dim)
        self.jointstylenet = JointStyleNet(1024, 1024)
        self.refencoder = ReferenceEncoder(hid_C = int(reference_dim * 2), out_C = 1024)
        
    def forward(self, encoder, decoder, reference, if_pro = True):
        
        if if_pro == True:
            
            encoder_out = self.encoder_proj(encoder)
            decoder_out = self.decoder_proj(decoder)
        else:
            encoder_out = encoder
            decoder_out = decoder
        reference_out = self.refencoder(reference)
        
        logit = encoder_out + decoder_out
        
        logit = self.jointstylenet(logit, reference_out)
        logit = self.output_linear(self.tanh(logit))

        # logit = F.log_softmax(logit, -1)
        # logit = self.hidden_liner(self.tanh(logit))
        # logit = self.output_linear(self.relu(logit))
        return logit

class Stage1Net(nn.Module):
    def __init__(self,
                 text_dim:int = 384,
                 num_vocabs:int = 513,
                 num_phonemes:int = 512,
                 token_dim:int = 512,
                 hid_token_dim:int = 512,
                 inner_dim:int = 512,
                 ref_dim:int = 512,
                 layer_nums= 6):
        super().__init__()
        self.textencoder = TextEncoder(text_dim,
                                       num_phonemes,
                                       layer_nums = layer_nums)
        self.tokenencoder = PredictionEncoder(token_dim,
                                              hid_token_dim,
                                              num_vocabs,
                                              )
        self.refencoder = ReferenceEncoder(hid_C = int(text_dim * 2), out_C = text_dim)
        self.jointer = JointNet()
        encoder_dim = 384
        decoder_dim = 512
        self.simple_token_proj = nn.Linear(decoder_dim, num_vocabs)
        self.simple_phone_proj = nn.Linear(encoder_dim, num_vocabs)
        self.fc = nn.Linear(int(text_dim * 2), text_dim)
    def forward(self,
                text_seq: Tensor,
                token_seq: Tensor,
                text_lens: Tensor,
                token_lens: Tensor,
                reference_audio: Tensor,
                true_seq: Tensor,
                lm_scale: float = 0.0,
                am_scale: float = 0.0,
                prune_range: int = 50,
                warmup: float = 1.0,
                reduction: str = 'none',
                
                ):
        text_seq = self.textencoder(text_seq, text_lens, warmup)
        
        token_seq, _  = self.tokenencoder(token_seq)
        
        boundary = torch.zeros((text_seq.size(0), 4), dtype = torch.int64, device = text_seq.device)
        boundary[:, 2] = token_lens.long()
        boundary[:, 3] = text_lens.long()
        lm = self.simple_token_proj(token_seq)
        am = self.simple_phone_proj(text_seq)
        
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad)  = k2.rnnt_loss_smoothed(
                lm = lm.float(),
                am = am.float(),
                symbols = true_seq,
                termination_symbol = 0,
                lm_only_scale = lm_scale,
                am_only_scale = am_scale,
                return_grad = True,
                reduction = 'sum'
            )
        ranges = k2.get_rnnt_prune_ranges(
            px_grad = px_grad,
            py_grad = py_grad,
            boundary = boundary,
            s_range = prune_range,
        )
        
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am = self.jointer.encoder_proj(text_seq),
            lm = self.jointer.decoder_proj(token_seq),
            ranges = ranges
        )
        
        logits = self.jointer(am_pruned, lm_pruned, reference_audio, if_pro = False)
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits = logits.float(),
                symbols = true_seq,
                ranges = ranges,
                termination_symbol = 0,
                boundary = boundary,
                reduction = 'sum'
            )
        
        
        return simple_loss, pruned_loss
    
    @torch.no_grad()
    def decode(self, 
               text_outputs: Tensor,
               reference_audio: Tensor,
               decode_type:str = 'top_k',
               top_k:int = 10,
               max_token_lens: int = 2048,
               ):
        batch = text_outputs.size(0)
        y_hats = list()
        targets = torch.LongTensor([0] * batch).to(text_outputs.device)
        targets = targets.unsqueeze(-1)
        time_num = 0
        h_0 = torch.zeros(2, text_outputs.shape[0], 512).to(text_outputs.device)
        c_0 = torch.zeros(2, text_outputs.shape[0], 512).to(text_outputs.device)
        max_lens = text_outputs.shape[1]
        hidden = (h_0, c_0)
        
        for i in range(int(max_lens)):
            pred = -1
            while(pred != 0):
                label_output, hidden = self.tokenencoder.inference(targets, hidden)
                text_output = text_outputs[:, i, :].unsqueeze(1)
                
                output = self.jointer(text_output, label_output, reference_audio)
                output = F.softmax(output, dim = -1)
                output = output.squeeze(1).squeeze(1)
                
                if decode_type == 'top_k':
                    top_k_output_values, top_k_output_indices = torch.topk(output, k = top_k, dim = -1)
                    
                    normed_top_k_output_values = F.softmax(top_k_output_values, dim = -1)
                
                    choosed_indices = torch.multinomial(normed_top_k_output_values, num_samples=1)
                
                    targets = top_k_output_indices[0, choosed_indices]
                    pred = targets

                elif decode_type == 'max':
                    targets = torch.argmax(output, dim = -1)
                    targets = targets.unsqueeze(0)
                    pred = targets
                
                time_num += 1
                if pred == 0:
                    break
                else:
                    y_hats.append(targets[0,:])
                if time_num >max_token_lens:
                    break
            if time_num > max_token_lens:
                break
        y_hats = torch.stack(y_hats, dim = 1)
        
        return y_hats
    @torch.no_grad()
    def draw_pic(self,
                text_seq: Tensor,
                token_seq: Tensor,
                text_lens: Tensor,
                token_lens: Tensor,
                reference_audio: Tensor,
                true_seq: Tensor,
                lm_scale: float = 0.25,
                am_scale: float = 0.0,
                prune_range: int = 50,
                warmup: float = 1.0,
                reduction: str = 'none'):
        
        text_seq = self.textencoder(text_seq, text_lens, warmup)
        token_seq, _  = self.tokenencoder(token_seq)
        
        boundary = torch.zeros((text_seq.size(0), 4), dtype = torch.int64, device = text_seq.device)
        boundary[:, 2] = token_lens
        boundary[:, 3] = text_lens
        lm = self.simple_token_proj(token_seq)
        am = self.simple_phone_proj(text_seq)
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad)  = k2.rnnt_loss_smoothed(
                lm = lm.float(),
                am = am.float(),
                symbols = true_seq,
                termination_symbol = 0,
                lm_only_scale = lm_scale,
                am_only_scale = am_scale,
                return_grad = True,
                reduction = reduction
            )
        ranges = k2.get_rnnt_prune_ranges(
            px_grad = px_grad,
            py_grad = py_grad,
            boundary = boundary,
            s_range = prune_range,
        )
        
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am = self.jointer.encoder_proj(text_seq),
            lm = self.jointer.decoder_proj(token_seq),
            ranges = ranges
        )

        logits = self.jointer(am_pruned, lm_pruned, reference_audio, if_pro = False)
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits = logits.float(),
                symbols = true_seq,
                ranges = ranges,
                termination_symbol = 0,
                boundary = boundary,
                reduction = reduction
            )
        
        B = px_grad.size(0)
        S = px_grad.size(1)
        T = px_grad.size(2) - 1

        px_grad_pad = torch.zeros(
            (B, 1, T + 1), dtype = px_grad.dtype, device = px_grad.device
        )
        py_grad_pad = torch.zeros(
            (B, S + 1, 1), dtype = px_grad.dtype, device = px_grad.device
        )
        px_grad_padded = torch.cat([px_grad, px_grad_pad], dim = 1)
        py_grad_padded = torch.cat([py_grad, py_grad_pad], dim = 2)
        tot_grad = px_grad_padded + py_grad_padded

        return simple_loss, pruned_loss

    @torch.no_grad()
    def recognize(self, inputs, input_lens, reference_audio):
        text_outputs = self.textencoder(inputs, input_lens)
        return self.decode(text_outputs, reference_audio, 'top_k')
