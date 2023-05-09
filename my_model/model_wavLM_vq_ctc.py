#直接用1024做的 gst长度用wavLM拼接 正确的 有VQ 有ctc 无grl 用的melgan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import pylab
import matplotlib.pyplot as plt
import requests
import torchaudio
from functools import reduce
from hparams import hparams as hp
from my_model import cbhg, vq, GST
from seqloss import sequence_mask
#from my_model import GST
from my_model.masked_layers import *
import pdb
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=512, kernel_size=5):
        super(ConvBlock, self).__init__()
        #Each convolution block contains 3 Conv1d layers with skip connection, and each Conv1d layer combines with IN and the ReLU
        #The kernel size, channel, stride of Conv1d layer are set to 5, 256 and 1,
        self.layers = nn.Sequential(ConvNorm(in_channels, out_channels, kernel_size=kernel_size,
                                             padding=kernel_size//2, w_init_gain='relu'),
                                    nn.InstanceNorm1d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvResnet(nn.Module):
    #The kernel size, channel, stride of Conv1d layer are set to 5, 256 and 1,
    def __init__(self, downsample, in_channels, out_channels=256):
        super(ConvResnet, self).__init__()
        self.downsample = downsample
        self.head = ConvBlock(in_channels, out_channels=out_channels, kernel_size=5)
        self.blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        for i, ds in enumerate(downsample):
            block = nn.Sequential(ConvBlock(in_channels=out_channels, 
                                            out_channels=out_channels, kernel_size=5),
                                  ConvBlock(out_channels, out_channels, kernel_size=5))
            if ds > 1:
                block.add_module(str(len(block)), nn.MaxPool1d(kernel_size=ds, stride=ds))
            self.blocks.append(block)
            self.merge_blocks.append(ConvBlock(out_channels, out_channels, kernel_size=1))
    
    def forward(self, input):
        x = self.head(input)
        for idx, (block, mblock) in enumerate(zip(self.blocks, self.merge_blocks)):
            ds = self.downsample[idx]
            if ds > 1:
                x = block(x) + F.max_pool1d(x, kernel_size=ds, stride=ds)
            else:
                x = block(x) + x
            x = mblock(x)
        return x


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self):
        super(Encoder, self).__init__()
        #4 convolution blocks and a bottleneck linear layer.
        self.convolutions = ConvResnet(downsample=[1,1,1,1], in_channels=1024, out_channels=256)
        
        #self.lstm = DynamicLSTM(256, 128, 2, batch_first=True, bidirectional=True)
        #vq
        self.proj = nn.Linear(256, hp.vq_embed_dim, bias=True)
        #ctc in 64_dim, out 40_dim
        self.ctc_rescon = CTCrescon(feat_dim=hp.vq_embed_dim, ctc_dim=40)

        if hp.use_EMA_vq:
            self.vq_codes = vq.VectorQuantizerEMA(hp.vq_num_embed, hp.vq_embed_dim, hp.commitment_cost, hp.decay)
        else:
            #wl vq_codes in 40_dim, out 64_dim
            self.vq_codes = vq.VectorQuantizer(hp.vq_num_embed, hp.vq_embed_dim, hp.commitment_cost)

        self.gst = GST.GST()

    #The decoder contains 3 Conv1d layers, a BLSTM layer, and a linear layer.
        dec_convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm((hp.vq_embed_dim+256) if i ==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'))
            dec_convolutions.append(conv_layer)
        self.decoder_convolutions = nn.ModuleList(dec_convolutions)

        self.dec_lstm = DynamicLSTM(512, 512, 1, batch_first=True, bidirectional=True)
        self.dec_proj = nn.Linear(1024, 80, bias=False)
    #       (mel_data, mel_data, wav2vec_data, seq_lengths)
    def forward(self, ref_mel, wav2vec_data, seq_lengths=None):
        ###wl 加上content encoder后进行 gst长度用wavLM拼接
        new_time_len = wav2vec_data.shape[1]
        # # N x C x T
        ###content encoder：
        wav2vec_data = wav2vec_data.permute(0, 2, 1)     ###[16, 1024, 384]
        wav2vec_data = self.convolutions(wav2vec_data)      ##送到content encoder 四个1D_conv in_1024 out_256
        # # N x T x C
        wav2vec_data = wav2vec_data.permute(0, 2, 1)    #####[16, 384, 256]
        wav2vec_data = self.proj(wav2vec_data)      #####[16, 384, 64] 注意是64 linear层，为了送到VQ
        # #ctc_out = x
        ctc_out = self.ctc_rescon(wav2vec_data)    ####[16, 384, 40]
        vq_loss, x, vq_perplexity, _ = self.vq_codes(wav2vec_data)   ###[16, 384, 64]   送入VQ的结果       
        style_emb = self.gst(ref_mel, None).squeeze(dim=1)  ####[16, 256]
        style_emb = style_emb.unsqueeze(1).expand(-1, new_time_len ,-1)   ####[16, 384, 256] mel特征与gst融合，得到style_emb
        #torch.cat拼接
        x = torch.cat((x, style_emb), dim=-1)   ####[16, 384,256+64]
        x = x.permute(0, 2, 1)     ####[16, 256+1024, 384]
        for conv in self.decoder_convolutions:
            x = F.relu(conv(x))
        x = x.permute(0, 2, 1)    ####[16, 384, 512]
        # print("176 lines x:{}".format(x.shape))
        x, _ = self.dec_lstm(x, None)
        ###3.6改动
        # x = self.dec_proj(x)
        
        return x, vq_loss, vq_perplexity, ctc_out




    def vq_value(self, x, ref_mel, seq_lengths=None):
        time_len = x.shape[1]
        # N x C x T
        x = x.permute(0, 2, 1)
        #x = self.head_convbank(x)
        x = self.convolutions(x)
        # N x T x C
        x = x.permute(0, 2, 1)
        #x, _ = self.lstm(x, None, flatten_params=False)
        x = self.proj(x)
        #ctc_out = x
        ctc_out = self.ctc_rescon(x)
        vq_loss, x, vq_perplexity, vq_code = self.vq_codes(x)

        return vq_code[0]  
    
    def spk_embed(self, x, ref_mel, seq_lengths=None):      
        style_emb = self.gst(ref_mel, None).squeeze(dim=1)
        return style_emb 


    def enc_out(self, x, ref_mel, seq_lengths=None):
        time_len = x.shape[1]
        # N x C x T
        x = x.permute(0, 2, 1)
        #x = self.head_convbank(x)
        x = self.convolutions(x)
        # N x T x C
        x = x.permute(0, 2, 1)
        #x, _ = self.lstm(x, None, flatten_params=False)
        x = self.proj(x)
        #ctc_out = x
        #ctc_out = self.ctc_rescon(x)
        #vq_loss, x, vq_perplexity, _ = self.vq_codes(x)
        #
        #style_emb = self.gst(ref_mel, None).squeeze(dim=1)
        #style_emb = style_emb.unsqueeze(1).expand(-1, time_len ,-1)
        #x = torch.cat((x, style_emb), dim=-1)
        #
        #x = x.permute(0, 2, 1)
        #for conv in self.decoder_convolutions:
        #    x = F.relu(conv(x))
        #x = x.permute(0, 2, 1)
        #
        #x, _ = self.dec_lstm(x, None)
        #x = self.dec_proj(x)

        #return x, vq_loss, vq_perplexity, ctc_out
        return x

class CTCrescon(nn.Module):
    # The CTC auxiliary network contains 3 Conv1d layers, a BLSTM layer and a linear layer
    def __init__(self, feat_dim, ctc_dim, n_convs=3, conv_channels=256,
                 n_bigru=1, hidden_size=128, proj_dim=256):
        super(CTCrescon, self).__init__()
        self.convs = nn.ModuleList()
        # Each Conv1 layer is combined with batch normalization and a ReLU activation layer   ###relu  ###原来的是relu    nn.Relu(inplace=True))
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
        #The number of BLSTM layer and linear layer units are set to 128 and 40.
        self.gru = DynamicGRU(input_size=conv_channels, hidden_size=hidden_size, num_layers=n_bigru,
                            batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(LinearNorm(in_dim=hidden_size*2, out_dim=ctc_dim))

    def forward(self, x, seq_lengths=None):
        batch_size = x.shape[0]
        # input: N x T x C_in, tranpose转置 to N x C_in x T
        x = x.permute(0, 2, 1)
        # N x C_out x T
        for conv_layer in self.convs:
            x = conv_layer(x)
        # N x T x C_out
        x = x.permute(0, 2, 1)
        # N x T x 2H
        x, _ = self.gru(x, seq_lengths, flatten_params=False)
        out = self.proj(x)
        return out
