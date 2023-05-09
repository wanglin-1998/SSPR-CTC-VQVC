#! /usr/bin/python
"""
 This module mainly get the linear spectrogram of the audio
"""
import os, sys, glob
import numpy as np
#sys.path.append("../scripts")
from multiprocessing import Pool
#from linearprocessing import get_linear
import audio 

condition = {
    "sample_rate": 16000,
    "hop_size": 160,  # 256=16ms  200=12.5ms, 160=10ms 
    "win_size": 640,  #1024=16ms 800=12.5ms, 640=10ms
    "n_fft": 1024,
    "num_linears": 80,

    "rescale": False,
    "rescaling_max": 0.999,

    "trim_silence": False,
    "trim_fft_size": 512,
    "trim_hop_size": 128,
    "trim_top_db": 23,

    "use_lws": False,

    "clip_linears_length": False,
    "max_linear_frames": 1300,

    "signal_normalization": True,
    "allow_clipping_in_normalization": False,
    "symmetric_linears": False,
    "symmetric_mels": False,
    "max_abs_value": 4.0,

    "min_level_db": -100,
    "ref_level_db": 20,

    "power": 1.2, 
    "griffin_lim_iters": 60
}

def linear2wav(linear_npy_dir, wav_dir):
        
    linear_file_list = os.listdir(linear_npy_dir)    
    for filename in linear_file_list:
        file_id = filename.split('.')[0]
        print(file_id)
        linear_file = os.path.join(linear_npy_dir, filename)
        wav_file = os.path.join(wav_dir, file_id + '.wav')
        linear = np.load(linear_file)
        linear = linear.T
        wav = audio.inv_linear_spectrogram(linear, condition)
        audio.save_wav(wav, wav_file, condition['sample_rate'])

if __name__ == "__main__":
    #linear_npy_dir = './temp_test/10ms/linear' 
    #wav_dir = './temp_wav/10ms/linear' 
    
    #linear_npy_dir = './temp_test/12.5ms/linear' 
    #wav_dir = './temp_wav/12.5ms/linear' 
   
    linear_npy_dir = sys.argv[1]
    wav_dir = sys.argv[2]
    
    os.makedirs(wav_dir, exist_ok=True)
    linear2wav(linear_npy_dir, wav_dir)
