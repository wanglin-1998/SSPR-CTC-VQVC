#3.6改动
import os
import os.path as osp
import sys
import argparse
from datetime import datetime
import math
import subprocess
import time
import traceback
import pdb
import torch
import numpy as np
import glob
import random
from my_model import model_wavLM_vq_ctc
from hparams import hparams as hp
import pdb
def get_ont_hot_embedding(style_id):
    nb_classes = hp.spk_num 
    target = style_id
    one_hot_target = np.eye(nb_classes, dtype=np.float32)[target]

    return one_hot_target

def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
    return x

def main():
    global device
    device = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/output_wavLM+vq+ctc/wavLM+vq+ctc_slt/model/model_12000.pt')
    parser.add_argument('--input_dir', default='/ssdhome/wl/data/vctk_vad_wav/finetune_data/mel')
    parser.add_argument('--input_wav2vec_dir', default='/ssdhome/wl/data/vctk_vad_wav/wavLM/rms_embed')
    parser.add_argument('--src_spk', default='rms40')
    parser.add_argument('--tgt_spk', default='slt100')
    parser.add_argument('--outdir', default='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/wavLM+vq+ctc_npy')

    args = parser.parse_args()

    model_path = args.model
    input_dir = args.input_dir
    input_wav2vec_dir = args.input_wav2vec_dir
    src_spk = args.src_spk
    tgt_spk = args.tgt_spk
    outdir = args.outdir

    mel_posnet_outdir = outdir
    os.makedirs(mel_posnet_outdir, exist_ok=True)

    print(torch.cuda.is_available())
    
    torch.cuda.current_device()
    torch.cuda._initialized = True
    
    use_cuda = hp.use_cuda and torch.cuda.is_available()
    assert use_cuda == True
    print("line 55 {}".format(device))
    #print(torch.cuda.current_device())
    print("device: ", device)

    torch.cuda.current_device()
    torch.cuda._initialized = True
    model = model_wavLM_vq_ctc.Encoder().to(device)

    #ckpt_model = torch.load(model_path)
    ckpt_model = torch.load(model_path, map_location='cpu')
    if "state_dict" in ckpt_model:
        model.load_state_dict(ckpt_model["state_dict"])
    else:
        model.load_state_dict(ckpt_model)
    model.eval()

    src_mel_dir = os.path.join(input_dir, src_spk)
    input_file_list = os.listdir(src_mel_dir)    ### mel/p225/   mel/p227

    #src_wav2vec_dir = os.path.join(input_wav2vec_dir, src_spk)
    #input_wav2vec_list = os.listdir(input_wav2vec_dir)   ####  wav2vec2.0/embed_right/p225+p227
    ###????
    ref_mel_files = sorted(os.listdir(os.path.join(input_dir, tgt_spk)))    ####    mel/p226/   mel/p228
    ###???osp
    ref_mel = np.load(os.path.join(input_dir, tgt_spk, ref_mel_files[0]))  #### mel/p226/***.npy
    ref_mel = torch.FloatTensor(ref_mel).unsqueeze(0).to(device)
    ref_input_lengths = torch.LongTensor([ref_mel.shape[1]]).to(device)

    input_file_list.sort()
    for in_file in input_file_list:  ##?? src?mel p225_011.npy
        print(in_file)
        file_id = in_file.split('.')[0] ##p225_011
        # in_file_path = os.path.join(src_mel_dir, in_file)  ##mel/p225/p225_011.npy
        # input_mel = np.load(in_file_path)

        # #in_wav2vec_data_path = os.path.join(src_wav2vec_dir, in_file)  ####  wav2vec2.0/embed_right/p225/p225_011.npy
        # in_wav2vec_data_path = os.path.join(input_wav2vec_dir, in_file)
        # wav2vec_data = np.load(in_wav2vec_data_path)

        # n_frames = input_mel.shape[0]
        # padded_length = max(400, n_frames) + hp.freq - max(400, n_frames) % hp.freq
        # input_mel = _pad_2d(input_mel, padded_length)

        # input_mel = torch.FloatTensor(input_mel).unsqueeze(0)
        # wav2vec_data = _pad_2d(wav2vec_data, padded_length)

        # wav2vec_data = torch.FloatTensor(wav2vec_data).unsqueeze(0)
        # input_lengths = torch.LongTensor([n_frames])
        # # print(input_lengths)

        # input_mel = input_mel.to(device)

        # seq_lengths = input_lengths.to(device) # None for unpacking
        # # print(input_lengths)

        # # input_mel = input_mel.to(device)

        # seq_lengths = input_lengths.to(device) # None for unpacking
        # # TODO: measure accuracy of speaker classifier
        # #æœ‰åŒºåˆ«wl
        # (mel_outputs, vq_loss, vq_perplexity, _) = model(ref_mel, wav2vec_data, seq_lengths)  #torch.Size([1, 588, 80])
        # #mel_outputs= model(input_mel, ref_mel, wav2vec_data, seq_lengths)
        
        # #mel_outputs = mel_outputs[0][:n_frames1, :]     

        # mel_outputs = mel_outputs[0][:n_frames, :]
        # mel_outputs_np = mel_outputs.cpu().data.numpy()

        # mel_file = os.path.join(mel_posnet_outdir, file_id + '_mel')
        # np.save(mel_file, mel_outputs_np)

        # print('predict {0} done'.format(file_id))

        in_file_path = os.path.join(src_mel_dir, in_file)  ##mel/p225/p225_011.npy
        input_mel = np.load(in_file_path)

        #in_wav2vec_data_path = os.path.join(src_wav2vec_dir, in_file)  ####  wav2vec2.0/embed_right/p225/p225_011.npy
        in_wav2vec_data_path = os.path.join(input_wav2vec_dir, in_file)
        wav2vec_data = np.load(in_wav2vec_data_path)

        n_frames = input_mel.shape[0]
        n_frames1 = wav2vec_data.shape[0]

        padded_length = max(400, n_frames1) + hp.freq - max(400, n_frames1) % hp.freq

        # input_mel = _pad_2d(input_mel, padded_length)
        # input_mel = torch.FloatTensor(input_mel).unsqueeze(0)

        wav2vec_data = _pad_2d(wav2vec_data, padded_length)
        wav2vec_data = torch.FloatTensor(wav2vec_data).unsqueeze(0)
        #gai    
        input_lengths = torch.LongTensor([n_frames1])
        # print(input_lengths)

        # input_mel = input_mel.to(device)

        seq_lengths = input_lengths.to(device) # None for unpacking
        # TODO: measure accuracy of speaker classifier
        #æœ‰åŒºåˆ«wl
        #(mel_outputs, _) = model(ref_mel, wav2vec_data, seq_lengths)
        (mel_outputs, vq_loss, vq_perplexity, _) = model(ref_mel, wav2vec_data, seq_lengths)
        mel_outputs = mel_outputs[0][:n_frames1, :]
        mel_outputs_np = mel_outputs.cpu().data.numpy()

        mel_file = os.path.join(mel_posnet_outdir, file_id + '_mel')
        np.save(mel_file, mel_outputs_np)

        print('predict {0} done'.format(file_id))

if __name__ == '__main__':
    main()
