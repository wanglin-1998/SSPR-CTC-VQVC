# SSPR-CTC-VQVC
本项目是引入预训练表示混合矢量量化和CTC的语音转换的源码。包括训练阶段和转换阶段。
1、训练阶段先用VCTK数据集训到30w步，再用CMU数据集进行微调。分别使用train_wavLM_VQ_ctc.py和finetune_wavLM_VQ_ctc.py
2、转换阶段使用inference_wavLM_VQ_ctc.py生成mel谱。
3、用util包下的文件生成语音。
