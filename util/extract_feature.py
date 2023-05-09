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

def run_mel(args):
    wav_file = args[0]
    mel_file = args[1]
    wav = audio.load_wav(wav_file, condition['sample_rate'])
    mel = audio.melspectrogram(wav, condition)
    mel = mel.T
    np.save(mel_file, mel)

def get_mel_spectrogram(wav_dir, mel_dir):
    spk_list = os.listdir(wav_dir)
    for spk in spk_list:
        spk_wav_dir = os.path.join(wav_dir, spk)
        spk_mel_dir = os.path.join(mel_dir, spk)
        os.makedirs(spk_mel_dir, exist_ok=True)
        
        wav_file_list = os.listdir(spk_wav_dir)
        
        pool = Pool(10)
        for filename in wav_file_list:
            file_id = filename.split('.')[0]
            wav_file = os.path.join(spk_wav_dir, filename)
            mel_file = os.path.join(spk_mel_dir, file_id)
            #run_mel(wav_file, mel_file)
            args = (wav_file, mel_file)     
            result = pool.apply_async(run_mel, args=(args,))
            #print(result.get())
            print(file_id)
        pool.close()
        pool.join()

def run_linear(args):
    wav_file = args[0]
    linear_file = args[1]
    wav = audio.load_wav(wav_file, condition['sample_rate'])
    linear = audio.linearspectrogram(wav, condition)
    linear = linear.T
    np.save(linear_file, linear)

def get_linear_spectrogram(wav_dir, linear_dir):
    spk_list = os.listdir(wav_dir)
    for spk in spk_list:
        spk_wav_dir = os.path.join(wav_dir, spk)
        spk_linear_dir = os.path.join(linear_dir, spk)
        os.makedirs(spk_linear_dir, exist_ok=True)
        
        wav_file_list = os.listdir(spk_wav_dir)
        
        pool = Pool(20)
        for filename in wav_file_list:
            file_id = filename.split('.')[0]
            wav_file = os.path.join(spk_wav_dir, filename)
            linear_file = os.path.join(spk_linear_dir, file_id)
            #run_linear(wav_file, linear_file)
            args = (wav_file, linear_file)     
            result = pool.apply_async(run_linear, args=(args,))
            #print(result.get())
            print(file_id)
        pool.close()
        pool.join()

if __name__ == "__main__":
    wav_dir = '/home7/wjc505/ctc-vc/data/cmu_wav' 
    
    mel_dir = '/home7/wjc505/ctc-vc/data/cmu_mel' 
    
    os.makedirs(mel_dir, exist_ok=True)
    #os.makedirs(linear_dir, exist_ok=True)
    get_mel_spectrogram(wav_dir, mel_dir)
    #get_linear_spectrogram(wav_dir, linear_dir)
