import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from hparams import hparams as hp
from my_model import cbhg
#from gradient_reversal_layer import GradientReversal
from seqloss import sequence_mask
#from aux_layers import cycle_shift
from my_model import GST
from my_model.masked_layers import *
import pdb


class ConvBank(nn.Module):
    def __init__(self, in_channels, out_channels, k_min=1, k_max=16, k_interval=1):
        super(ConvBank, self).__init__()
        self.conv_banks = nn.ModuleList()
        for k in range(k_min, k_max + 1, k_interval):
            layer = nn.Sequential(ConvNorm(in_channels, in_channels, kernel_size=k, padding=k//2, w_init_gain='relu'),
                                  nn.InstanceNorm1d(in_channels), 
                                  nn.ReLU(inplace=True))
            self.conv_banks.append(layer)
        in_c = in_channels * ((k_max - k_min + 1) // k_interval) 
        self.proj = ConvNorm(in_c, out_channels, kernel_size=1)

    def forward(self, x):
        out = [conv(x) for conv in self.conv_banks]
        out = torch.cat(out, dim=1)
        out = self.proj(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=512, bneck_channels=256, kernel_size=5):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(ConvNorm(in_channels, bneck_channels, kernel_size=1, w_init_gain='relu'), 
                                    nn.InstanceNorm1d(bneck_channels), 
                                    nn.ReLU(inplace=True),
                                    ConvNorm(bneck_channels, out_channels, kernel_size=kernel_size, 
                                             padding=kernel_size//2, w_init_gain='relu'), 
                                    nn.InstanceNorm1d(out_channels), 
                                    nn.ReLU(inplace=True))
    
    def forward(self, x):
        out = self.layers(x)
        return out


class ConvResnet(nn.Module):
    def __init__(self, in_channels, out_channels=512, bneck_channels=256, n_blocks=6):
        super(ConvResnet, self).__init__()
        self.head = ConvBlock(in_channels, out_channels=384, bneck_channels=192, kernel_size=11)
        self.skip_conv = ConvNorm(384, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = nn.Sequential(ConvBlock(in_channels=384 if i==0 else out_channels, 
                                            out_channels=out_channels, bneck_channels=bneck_channels, kernel_size=5),
                                  ConvBlock(in_channels, out_channels, bneck_channels, kernel_size=5))
            self.blocks.append(block)
    
    def forward(self, input):
        x = self.head(input)
        for idx, block in enumerate(self.blocks):
            if idx == 0:
                x = block(x) + self.skip_conv(x)
            else:
                x = block(x) + x
        return x
        

      
class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.InstanceNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        #self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)
        self.lstm = DynamicLSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org, seq_lengths=None):
        time_len = x.shape[1]
        c_org = c_org.unsqueeze(dim=1).expand(-1, time_len, -1)
        x = torch.cat((x, c_org), dim=-1)
        if seq_lengths is not None:
            # N x T x 1
            mask = sequence_mask(seq_lengths, time_len).unsqueeze(dim=-1)
            x = x * mask
        # N x C x T
        x = x.permute(0, 2, 1)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        # N x T x C
        x = x.permute(0, 2, 1)

        outputs, _ = self.lstm(x, seq_lengths, flatten_params=True)

        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))
        return codes, outputs
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = DynamicLSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.InstanceNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = DynamicLSTM(dim_pre, 1024, 1, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, hp.num_mels)

    def forward(self, x, seq_lengths=None):
        padded_len = x.shape[1]
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x, seq_lengths, flatten_params=False)

        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x, seq_lengths, flatten_params=False)
        
        decoder_output = self.linear_projection(outputs)
        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.InstanceNorm1d(512))
        )
        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='relu'),
                    nn.InstanceNorm1d(512))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, hp.num_mels,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'))
            )

    def forward(self, x, seq_lengths=None):
        for i in range(len(self.convolutions) - 1):
            x = F.relu(self.convolutions[i](x))
        x = self.convolutions[-1](x)
        return x    


class SpeakerClassifier(nn.Module):
    def __init__(self, feat_dim, target_classes, n_convs=3, conv_channels=256,
                 n_bigru=1, hidden_size=128, proj_dim=256):
        super(SpeakerClassifier, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(n_convs):
            layer = nn.Sequential(
                ConvNorm(feat_dim if i==0 else conv_channels,
                         conv_channels,
                         kernel_size=5, 
                         stride=1,
                         padding=2,
                         dilation=1, 
                         w_init_gain='relu'),
                nn.BatchNorm1d(conv_channels), 
                nn.ReLU(inplace=True))
            self.convs.append(layer)
        
        self.hidden_size = hidden_size
        self.n_bigru = n_bigru
        self.gru = DynamicGRU(input_size=conv_channels, hidden_size=hidden_size, num_layers=n_bigru,
                            batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(LinearNorm(in_dim=2 * hidden_size, out_dim=proj_dim, w_init_gain='relu'), 
                                  nn.BatchNorm1d(proj_dim), 
                                  nn.ReLU(inplace=True), 
                                  LinearNorm(in_dim=proj_dim, out_dim=target_classes))
        
    def forward(self, x, seq_lengths=None):
        batch_size = x.shape[0]
        # input: N x T x C_in, tranpose to N x C_in x T
        x = x.permute(0, 2, 1)
        # N x C_out x T
        for conv_layer in self.convs:
            x = conv_layer(x)
        # N x T x C_out
        x = x.permute(0, 2, 1)
        # N x T x 2H
        _, x = self.gru(x, seq_lengths, flatten_params=False)
        # get the last output of each direction, and then concat: N x 2H
        x = x.view(self.n_bigru, 2, batch_size, self.hidden_size)
        x = x[-1].permute(1, 0, 2).contiguous().view(batch_size, -1)
        # out logits: N x nclasses
        out = self.proj(x)
        return out


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        self.c_linear_layer = torch.nn.Linear(hp.embed_dim, dim_emb, bias=True)
        self.gst = GST.GST()
        self.out_linear_layer = torch.nn.Linear(160, hp.num_linear, bias=True)

        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.cbhg = cbhg.CBHG(hp.num_mels, K=16, projections=[128, 128])

        self.middle_net = DynamicLSTM(dim_neck * 2, dim_neck, 3, batch_first=True, bidirectional=True)

        self.dim_emb = dim_emb
        self.dim_neck = dim_neck

    def forward(self, mel_data, content_vectors, ref_mel_data, seq_lengths=None, ref_seq_lengths=None):
        batch_size, time_len = tuple(mel_data.shape[:2])
    
        c_style_emb = self.c_linear_layer(content_vectors)
        s_style_emb = self.gst(ref_mel_data, ref_seq_lengths).squeeze(dim=1)
        
        codes, full_encoder_out = self.encoder(mel_data, c_style_emb, seq_lengths)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(time_len/len(codes)),-1))
        sampled_encoder_out = torch.cat(tmp, dim=1)
        sampled_encoder_out, _ = self.middle_net(sampled_encoder_out, seq_lengths)
        
        cross_mel_outputs, cycle_mel_outputs = None, None
        mask, mask_rep, seq_lengths_rep = None, None, None
        if seq_lengths is not None:
            mask = sequence_mask(seq_lengths, time_len).unsqueeze(dim=-1)
            # repeat interleave
            seq_lengths_rep = seq_lengths.unsqueeze(dim=-1).repeat(1, 2).view(-1)
            mask_rep = sequence_mask(seq_lengths_rep, time_len).unsqueeze(dim=-1)

        if self.training and (hp.use_cross_mel or hp.use_cycle_loss):
            # repeat interleave to keep sorted sequence in order, e.g. (0, 0, 1, 1, 2, 2)
            code_exp = sampled_encoder_out.permute(1, 2, 0).unsqueeze(dim=-1).expand(-1, -1, -1, 2)
            code_exp = code_exp.contiguous().view(time_len, -1, 2 * batch_size).permute(2, 0, 1)

            # cycle shift down and cat interleave
            s_style_emb_shift = cycle_shift(s_style_emb, hp.shift)
            s_style_emb_cat = torch.cat((s_style_emb, s_style_emb_shift), dim=-1).view(2 * batch_size, -1)
            s_style_emb_exp = s_style_emb_cat.unsqueeze(dim=1).expand(-1, time_len, -1)

            c_style_emb_shift = cycle_shift(c_style_emb, hp.shift)
            c_style_emb_cat = torch.cat((c_style_emb, c_style_emb_shift), dim=-1).view(2 * batch_size, -1)

            encoder_outputs = torch.cat((code_exp, s_style_emb_exp), dim=-1)
            encoder_outputs = mask_tensor(encoder_outputs, mask_rep)

            cross_mel_outputs = self.decoder(encoder_outputs, seq_lengths_rep)
            # there are blstms in encoder, so reset padding part of output before input encoder again
            cross_mel_outputs = mask_tensor(cross_mel_outputs, mask_rep)
            
            mel_outputs = cross_mel_outputs[0::2]
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1), seq_lengths)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
            mel_outputs_postnet = mask_tensor(mel_outputs_postnet, mask)

            linear_outputs_postnet_temp = self.cbhg(mel_outputs_postnet.detach(), input_lengths=seq_lengths)
            linear_outputs_postnet = self.out_linear_layer(linear_outputs_postnet_temp)
            linear_outputs_postnet = mask_tensor(linear_outputs_postnet, mask)

            codes_org = torch.cat([x.unsqueeze(1) for x in codes], dim=1)
            # repeat interleave as code_exp
            codes_org = codes_org.permute(1, 2, 0).unsqueeze(dim=-1).expand(-1, -1, -1, 2)
            codes_org = codes_org.contiguous().view(-1, 2 * self.dim_neck, 2 * batch_size).permute(2, 0, 1)

            cross_codes, _ = self.encoder(cross_mel_outputs, c_style_emb_cat, seq_lengths_rep)
            codes_pred = torch.cat([x.unsqueeze(1) for x in cross_codes], dim=1)
            if hp.use_cycle_loss:
                tmp = []
                for code in cross_codes:
                    # get only cross pairs
                    tmp.append(code[1::2].unsqueeze(1).expand(-1, int(time_len/len(cross_codes)), -1))
                upsampled_cross_codes = torch.cat(tmp, dim=1)
                upsampled_cross_codes, _ = self.middle_net(upsampled_cross_codes, seq_lengths)

                s_style_emb_recon = s_style_emb.unsqueeze(dim=1).expand(-1, time_len, -1)
                cycle_encoder_outputs = torch.cat((upsampled_cross_codes, s_style_emb_recon), dim=-1)
                cycle_encoder_outputs = mask_tensor(cycle_encoder_outputs, mask)
                cycle_mel_outputs = self.decoder(cycle_encoder_outputs, seq_lengths)
        else:            
            encoder_outputs = torch.cat((sampled_encoder_out, s_style_emb.unsqueeze(1).expand(-1,mel_data.size(1),-1)), dim=-1)
            encoder_outputs = mask_tensor(encoder_outputs, mask)
            mel_outputs = self.decoder(encoder_outputs, seq_lengths)

            mel_outputs = mask_tensor(mel_outputs, mask)
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1), seq_lengths)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
            mel_outputs_postnet = mask_tensor(mel_outputs_postnet, mask)

            linear_outputs_postnet_temp = self.cbhg(mel_outputs_postnet.detach(), input_lengths=seq_lengths)
            linear_outputs_postnet = self.out_linear_layer(linear_outputs_postnet_temp)
            linear_outputs_postnet = mask_tensor(linear_outputs_postnet, mask)

            #TODO: [batch_size, frame*dim_neck*2] --> [batch_size, frame, dim_neck*2]
            codes_org = torch.cat([x.unsqueeze(1) for x in codes], dim=1)
            codes_pred, _ = self.encoder(mel_outputs, c_style_emb, seq_lengths)
            codes_pred = torch.cat([x.unsqueeze(1) for x in codes_pred], dim=1)
        
        return mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, full_encoder_out, \
                cross_mel_outputs, sampled_encoder_out, cycle_mel_outputs

class GeneratorWithSpeakerClassifierCont(nn.Module):
    """Generator network with GRL and speaker-classifier append to content codes"""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, grl_weight=1.0):
        super(GeneratorWithSpeakerClassifierCont, self).__init__()
        self.generator = Generator(dim_neck, dim_emb, dim_pre, freq)
        self.grl1 = GradientReversal(lambda_=grl_weight)
        self.spk_classifier1 = SpeakerClassifier(feat_dim=2*dim_neck, target_classes=hp.outdim_spkcls_cont)
        self.grl2 = GradientReversal(lambda_=grl_weight)
        self.spk_classifier2 = SpeakerClassifier(feat_dim=2*dim_neck, target_classes=hp.outdim_spkcls_cont)

    def forward(self, mel_data, content_vectors, speaker_vectors, seq_lengths=None, ref_seq_lengths=None):
        (mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, \
            full_encoder_out, _, sampled_encoder_out, cycle_mel_outputs)\
            = self.generator(mel_data, content_vectors, speaker_vectors, seq_lengths, ref_seq_lengths)
        full_speaker_feat = self.grl1(full_encoder_out)
        full_speaker_logits = self.spk_classifier1(full_speaker_feat, seq_lengths)
        sampled_speaker_feat = self.grl2(sampled_encoder_out)
        sampled_speaker_logits = self.spk_classifier2(sampled_speaker_feat, seq_lengths)
        return  mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, \
                 full_speaker_logits, sampled_speaker_logits, cycle_mel_outputs


class GeneratorWithSpeakerClassifierRecon(nn.Module):
    """Generator network with speaker-classifier append to decoder output (mel-spectrum)"""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(GeneratorWithSpeakerClassifierRecon, self).__init__()
        self.generator = Generator(dim_neck, dim_emb, dim_pre, freq)
        self.spk_classifier = SpeakerClassifier(feat_dim=hp.num_mels, target_classes=hp.outdim_spkcls_recon)

    def forward(self, mel_data, content_vectors, speaker_vectors, seq_lengths=None):
        (mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, full_encoder_out, cross_mel_outputs)\
            = self.generator(mel_data, content_vectors, speaker_vectors, seq_lengths)
        if self.training and hp.use_recon_speaker_classifier:
            speaker_logits = self.spk_classifier(cross_mel_outputs, seq_lengths)
        else:
            speaker_logits = self.spk_classifier(mel_outputs, seq_lengths)
        return  mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, speaker_logits


class GeneratorWithSpeakerClassifierBoth(nn.Module):
    """
        Generator network with speaker-classifier append to: 
        (a) content codes with GRL
        (b) decoder output (mel-spectrum)
    """ 
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, grl_weight=1.0):
        super(GeneratorWithSpeakerClassifierBoth, self).__init__()
        self.generator = Generator(dim_neck, dim_emb, dim_pre, freq)
        self.grl1 = GradientReversal(lambda_=grl_weight)
        self.spk_classifier1 = SpeakerClassifier(feat_dim=2*dim_neck, target_classes=hp.outdim_spkcls_cont)
        self.grl2 = GradientReversal(lambda_=grl_weight)
        self.spk_classifier2 = SpeakerClassifier(feat_dim=2*dim_neck, target_classes=hp.outdim_spkcls_cont)
        self.spk_classifier_recon = SpeakerClassifier(feat_dim=hp.num_mels, target_classes=hp.outdim_spkcls_recon)

    def forward(self, mel_data, content_vectors, speaker_vectors, seq_lengths=None):
        (mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, \
            full_encoder_out, _, sampled_encoder_out, cycle_mel_outputs)\
            = self.generator(mel_data, content_vectors, speaker_vectors, seq_lengths)
        full_speaker_feat = self.grl1(full_encoder_out)
        full_speaker_logits = self.spk_classifier1(full_speaker_feat, seq_lengths)
        sampled_speaker_feat = self.grl2(sampled_encoder_out)
        sampled_speaker_logits = self.spk_classifier2(sampled_speaker_feat, seq_lengths)

        seq_lengths_rep = None
        if seq_lengths is not None:
            seq_lengths_rep = seq_lengths.unsqueeze(dim=-1).repeat(1, seq_lengths.shape[0]).view(-1)
        if self.training and hp.use_recon_speaker_classifier:
            speaker_logits_recon = self.spk_classifier_recon(cycle_mel_outputs, seq_lengths_rep)
        else:
            speaker_logits_recon = self.spk_classifier_recon(mel_outputs, seq_lengths) 
        return  mel_outputs, mel_outputs_postnet, linear_outputs_postnet, codes_org, codes_pred, \
                    full_speaker_logits, sampled_speaker_logits, cycle_mel_outputs, speaker_logits_recon
