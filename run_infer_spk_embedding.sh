#!/bin/bash

# GPU_ID=0

#test_mel_dir='/home7/wjc505/ctc-vc/data/mel/'
test_mel_dir='/ssdhome/wl/data/vctk_vad_wav/mel'
#input_wav2vec_dir='/ssdhome/wl/data/vctk_vad_wav/wavLM/vctk_embed'
model='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/output_wavLM+vq+ctc/wavLM+vq+ctc/model/model_150000.pt'
#model='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_wl/speaker_1024d_new/1024d_vctk/model/model_300000.pt'
#model='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_wl/1024d_woctc+grl/woctc+grl_vctk/model/model_300000.pt'
all_spk='p225 p228 p229 p230 p231 p226 p227 p232 p237 p241'
save_dir=SSPR-CTC-VQVC
for spk in $all_spk; do
	python inference_spk_embed_wl.py \
    	--model=${model} \
    	--input_dir=${test_mel_dir} \
    	--spk=${spk} \
    	--outdir="${save_dir}/${spk}"
done
