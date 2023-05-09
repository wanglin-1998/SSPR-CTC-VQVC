#!/bin/bash

GPU_ID=0

test_mel_dir='/ssdhome/wl/data/vctk_vad_wav/finetune_data/mel'
wav2vec_dir='/ssdhome/wl/data/vctk_vad_wav/wavLM_withVCTK/rms_embed'
#model=output/bdl_base_ft/model/model_400.pt
#model=output/slt_base_ft/model/model_400.pt
model='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_wl/output_withpreVCTK_wavLM+ctc/withpreVCTK_wavLM_ctc_bdl_finetune/model/model_3000.pt'
#all_src_spk='clb20 rms20'
all_src_spk='rms40'
#all_dst_spk='bdl100'
#all_dst_spk='slt100 bdl100'
all_dst_spk='bdl100'
#save_dir=mel_vq40_base_slt
save_dir='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_wl/output_withpreVCTK_wavLM+ctc/embed_npy'
for src_spk in $all_src_spk; do
	for tgt_spk in $all_dst_spk; do
		#[ "$src_spk" == "$tgt_spk" ] && continue
		echo "$src_spk to $tgt_spk" 
        CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_wavLM_ctc.py \
        	--model=${model} \
        	--input_dir=${test_mel_dir} \
			--input_wav2vec_dir=${wav2vec_dir} \
        	--src_spk=${src_spk} \
        	--tgt_spk=${tgt_spk} \
        	--outdir="${save_dir}/${src_spk}.to.${tgt_spk}"
	done
done


# test_mel_dir='/ssdhome/wl/data/vctk_vad_wav/finetune_data/mel'
# model='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/output_wavLM/vctk_wavLM/model/model_540000.pt'
# all_src_spk='clb20 rms20'
# #all_dst_spk='bdl100'
# all_dst_spk='slt100 bdl100'
# #save_dir=mel_vq40_base_slt
# save_dir='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/wavLM_embed_npy/ori_model_54'
# for src_spk in $all_src_spk; do
# 	for tgt_spk in $all_dst_spk; do
# 		#[ "$src_spk" == "$tgt_spk" ] && continue
# 		echo "$src_spk to $tgt_spk" 

#         CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py \
#         	--model=${model} \
#         	--input_dir=${test_mel_dir} \
#         	--src_spk=${src_spk} \
#         	--tgt_spk=${tgt_spk} \
#         	--outdir="${save_dir}/${src_spk}.to.${tgt_spk}"
# 	done
# done