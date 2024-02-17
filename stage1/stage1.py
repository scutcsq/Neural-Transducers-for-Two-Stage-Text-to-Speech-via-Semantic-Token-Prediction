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
from ice_conformer.scaling import ScaledLinear, ScaledEmbedding
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

        self.embedding = ScaledEmbedding(n_text,
                                        in_dim)
        
        
    def forward(self, x, x_lens, warmup = 1.0):
        x = self.embedding(x)
        layer_results, x_lens = self.conformer(x, x_lens, warmup = warmup)
        encoder_out = layer_results[-1]
        encoder_out = F.relu(encoder_out)
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
        self.embedding = ScaledEmbedding(n_token, in_dim)
        self.inner_size = 513
        self.relu = nn.ReLU()
        self.fc = ScaledLinear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, self.inner_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        x = self.embedding(x)
        x, y  = self.lstm(x, (h0, c0))
        # x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
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
        x = self.relu(x)
        # x = self.fc(x)
        return x, hidden



## Reference Network
# self.ref_p = ReferenceEncoder(hid_C = hidden_channels, out_C = hidden_channels)


## Joint Net
class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        # affine = nn.Linear(in_dim, out_dim)
        affine = ScaledLinear(in_dim, out_dim)
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

class Conditional_LayerNorm(nn.Module):

    def __init__(self,
                normal_shape: int = 513,
                epsilon: int = 1e-5
                ):
        # from https://github.com/tuanh123789/AdaSpeech/blob/main/model/adaspeech_modules.py line 162
        super(Conditional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = 513
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
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1).unsqueeze(1)
        y += bias.unsqueeze(1).unsqueeze(1)

        return y


class JointStyleBlock(nn.Module):
    def __init__(self,
                 ref_size: int = 512,
                 audio_size: int = 512,
                 ):
        super().__init__()
        self.fc = ScaledLinear(ref_size, int(ref_size) * 2)
        self.fc2 = ScaledLinear(int(ref_size) * 2, ref_size)
        self.ln = StyleAdaptiveLayerNorm(ref_size, audio_size)
        # self.ln = Conditional_LayerNorm()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self, x, ref_audio):
        x1 = self.ln(x, ref_audio)
        x1 = self.fc(x1)
        x1 = self.relu(x1)
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
                 reference_dim:int = 512,
                 joint_dim:int = 512,
                 hidden_dim:int = 2048,
                 vocab_size: int = 513
                 ):
        super().__init__()
        self.encoder_proj = ScaledLinear(encoder_dim, vocab_size)
        self.decoder_proj = ScaledLinear(decoder_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.hidden_liner = ScaledLinear(joint_dim, hidden_dim)
        # self.output_linear = ScaledLinear(hidden_dim, vocab_size)
        self.output_linear = ScaledLinear(joint_dim, vocab_size)
        self.jointstylenet = JointStyleNet(vocab_size, reference_dim)
        self.refencoder = ReferenceEncoder(hid_C = int(reference_dim * 2), out_C = reference_dim)
    def forward(self, encoder, decoder, reference, if_pro = True):
        # print('jointnet: encoder: ', encoder.shape, ' decoder: ', decoder.shape)
        if if_pro == True:
            encoder_out = self.encoder_proj(encoder)
            decoder_out = self.decoder_proj(decoder)
        else:
            encoder_out = encoder
            decoder_out = decoder
        reference_out = self.refencoder(reference)
        if encoder_out.dim() == 3 and decoder_out.dim() == 3:
            seq_lens = encoder_out.size(1)
            tar_lens = decoder_out.size(1)
            
            encoder_out = encoder_out.unsqueeze(1)
            decoder_out = decoder_out.unsqueeze(1)

            encoder_out = encoder_out.repeat(1, tar_lens, 1, 1)
            decoder_out = decoder_out.repeat(1, 1, seq_lens, 1)
        logit = encoder_out + decoder_out
        # print('logits: ', logit.shape)
        # print('reference: ', reference_out.shape)
        logit = self.jointstylenet(logit, reference_out)
        logit = self.output_linear(self.relu(logit))
        logit = F.log_softmax(logit, -1)
        # logit = self.hidden_linear(self.tanh(logit))
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
        self.refencoder = ReferenceEncoder(hid_C = ref_dim, out_C = inner_dim)
        self.jointer = JointNet()
        encoder_dim = 384
        decoder_dim = 512
        self.simple_token_proj = ScaledLinear(decoder_dim, num_vocabs)
        self.simple_phone_proj = ScaledLinear(encoder_dim, num_vocabs)
    def forward(self,
                text_seq: Tensor,
                token_seq: Tensor,
                text_lens: Tensor,
                token_lens: Tensor,
                reference_audio: Tensor,
                true_seq: Tensor,
                lm_scale: float = 0.25,
                am_scale: float = 0.25,
                prune_range: int = 50,
                warmup: float = 1.0,
                reduction: str = 'none',
                
                ):
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
        
        
        return simple_loss, pruned_loss
    
    @torch.no_grad()
    def decode(self, 
               text_outputs: Tensor,
               max_lens: int,
               reference_emb: Tensor,
               max_token_lens: int = 2048):
        batch = text_outputs.size(0)
        y_hats = list()
        targets = torch.LongTensor([0] * batch).to(text_outputs.device)
        targets = targets.unsqueeze(-1)
        time_num = 0
        for i in range(int(max_lens)):
            pred = -1
            while(pred != 0):
                if time_num == 0:
                    label_output, hidden = self.tokenencoder.inference(targets)
                else:
                    label_output, hidden = self.tokenencoder.inference(targets, hidden)
                # print('result: ', hidden[0].shape)
                text_output = text_outputs[:, i, :].unsqueeze(1)
                # print('text_output: ', text_output.shape, 'label_output: ', label_output.shape)
                output = self.jointer(text_output, label_output, reference_emb)
                output = F.log_softmax(output, dim = -1)
                output = output.squeeze(1).squeeze(1)
                
                top_k_output_values, top_k_output_indices = torch.topk(output, k = 5, dim = -1)
                normed_top_k_output_values = F.log_softmax(top_k_output_values, dim = -1)
                choosed_indices = torch.multinomial(normed_top_k_output_values, num_samples=1)
                # print('choosed_indices: ', choosed_indices, choosed_indices.shape)
                targets = top_k_output_indices[0, choosed_indices]
                # print('targets:', targets, targets.shape)
                pred = targets
                # pred = output.max(1)[1]
                # targets = output.max(1)[1]
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
    def recognize(self, inputs, input_lens, reference_audio):
        text_outputs = self.textencoder(inputs)
        max_lens, _  = torch.max(input_lens, dim = -1)
        
        return self.decode(text_outputs, max_lens, reference_audio)
