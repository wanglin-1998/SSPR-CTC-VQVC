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
import torch.nn as nn
import numpy as np
import glob
import random
#from my_model import model_wavLM_ctc_grl
from my_model import model_wavLM_vq_ctc
#from my_model import model_vc_v2_yuanlai
from hparams import hparams as hp
import pdb
import collections

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
    device = "cuda:1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/output_wavLM+vq+ctc/wavLM+vq+ctc/model/model_150000.pt')
    parser.add_argument('--input_dir', default='/ssdhome/wl/data/vctk_vad_wav/mel')
    # parser.add_argument('--input_wav2vec_dir', default='/ssdhome/wl/data/vctk_vad_wav/wavLM/vctk_embed')
    parser.add_argument('--spk', default="")
    #'p225','p228','p229','p230','p231','p226','p227','p232','p237','p241'
    parser.add_argument('--outdir', default='convert_feature')

    args = parser.parse_args()

    model_path = args.model
    input_dir = args.input_dir
    # input_wav2vec_dir = args.input_wav2vec_dir
    spk = args.spk
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    # use_cuda = hp.use_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    model = model_wavLM_vq_ctc.Encoder().to(device)

    ckpt_model = torch.load(model_path)
    if "state_dict" in ckpt_model:
        model.load_state_dict(ckpt_model["state_dict"])
    else:
        model.load_state_dict(ckpt_model)
    model.eval()
    src_mel_dir = os.path.join(input_dir, spk)
    input_file_list = os.listdir(src_mel_dir)
    # input_wav2vec_dir = os.path.join(input_wav2vec_dir, spk)

    # ref_mel_files = sorted(os.listdir(os.path.join(input_dir, spk)))
    # ref_mel = np.load(osp.join(input_dir, spk, ref_mel_files[0]))
    # ref_mel = torch.FloatTensor(ref_mel).unsqueeze(0).to(device)
    # ref_input_lengths = torch.LongTensor([ref_mel.shape[1]]).to(device)

    input_file_list.sort()
    input_file_list = input_file_list[:10]
    for in_file in input_file_list:
        print(in_file)
        file_id = in_file.split('.')[0]
        in_file_path = os.path.join(src_mel_dir, in_file)
        input_mel = np.load(in_file_path)
        n_frames = input_mel.shape[0]
        padded_length = max(400, n_frames) + hp.freq - max(400, n_frames) % hp.freq
        input_mel = _pad_2d(input_mel, padded_length)

        input_mel = torch.FloatTensor(input_mel).unsqueeze(0)

        input_lengths = torch.LongTensor([n_frames])
        # print(input_lengths)

        input_mel = input_mel.to(device)

        seq_lengths = input_lengths.to(device) # None for unpacking
        # TODO: measure accuracy of speaker classifier
        spk_emb = model.spk_embed(input_mel, input_mel, seq_lengths)
        spk_emb_np = spk_emb.cpu().data.numpy()

        emb_file = os.path.join(outdir, file_id)
        np.save(emb_file, spk_emb_np)

        print('predict {0} done'.format(file_id))

if __name__ == '__main__':
    main()
