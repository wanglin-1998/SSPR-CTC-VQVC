#! /usr/bin/python
"""
 This module mainly get the mel spectrogram of the audio
"""
import os, sys, glob
import numpy as np
#sys.path.append("../scripts")
from multiprocessing import Pool
#from melprocessing import get_mel
import audio
import sys,os 

condition = {
    "sample_rate": 16000,
    "hop_size": 160,  # 256=16ms  200=12.5ms, 160=10ms 
    "win_size": 640,  #1024=16ms 800=12.5ms, 640=10ms
    "n_fft": 1024,
    "num_mels": 80,

    "rescale": False,
    "rescaling_max": 0.999,

    "trim_silence": False,
    "trim_fft_size": 512,
    "trim_hop_size": 128,
    "trim_top_db": 23,

    "use_lws": False,

    "clip_mels_length": False,
    "max_mel_frames": 1300,

    "signal_normalization": True,
    "allow_clipping_in_normalization": False,
    "symmetric_mels": False,
    "max_abs_value": 4.0,

    "min_level_db": -100,
    "ref_level_db": 20,

    "power": 1.2, 
    "griffin_lim_iters": 60
}

def mel2wav(mel_npy_dir, wav_dir):
        
    mel_file_list = os.listdir(mel_npy_dir)    
    for filename in mel_file_list:
        file_id = filename.split('.')[0]
        print(file_id)
        mel_file = os.path.join(mel_npy_dir, filename)
        wav_file = os.path.join(wav_dir, file_id + '.wav')
        mel = np.load(mel_file)
        mel = mel.T
        wav = audio.inv_mel_spectrogram(mel, condition)
        audio.save_wav(wav, wav_file, condition['sample_rate'])

if __name__ == "__main__":
    #mel_npy_dir = './temp_test/10ms/mel' 
    #wav_dir = './temp_wav/10ms/mel' 
    
    #mel_npy_dir = './temp_test/12.5ms/mel' 
    #wav_dir = './temp_wav/12.5ms/mel' 
   
    #mel_npy_dir = './temp_test/16ms/mel' 
    #wav_dir = './temp_wav/16ms/mel' 


    ###wl

    # mel_npy_dir = sys.argv[1]
    # wav_dir = sys.argv[2]
    mel_npy_dir ='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/wavLM+vq+ctc_npy'
    wav_dir = '/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/wavLM+vq+ctc_wav'
    # os.makedirs(wav_dir, exist_ok=True)
    
    spk_list = os.listdir(mel_npy_dir)
    for spk in spk_list: 
        spk_npy_dir = os.path.join(mel_npy_dir, spk)
        spk_wav_dir = os.path.join(wav_dir, spk)
        os.makedirs(spk_wav_dir, exist_ok=True)
        mel2wav(spk_npy_dir, spk_wav_dir)
