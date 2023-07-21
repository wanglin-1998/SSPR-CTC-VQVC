###改动一下 变成1024d model_wavLM_ctc_frame_copy##
import sys
import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import numpy as np
from datasets import feat_dataset_copy
from my_model import model_wavLM_vq_ctc
import tensorboard_logger
from tensorboard_logger import log_value
import seqloss as seqloss
from hparams_ft import hparams as hp
import pdb
import multiprocessing
multiprocessing.set_start_method('spawn',True)

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def norm_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**hp.lr_decay_power * np.minimum(step * warmup_steps**-(1 + hp.lr_decay_power), step**-hp.lr_decay_power)
    return lr

def step_learning_rate_decay(init_lr, global_step,
                             warmup_steps=4000,
                             anneal_rate=0.8,
                             anneal_interval=5000):
    # exponential指数_decay with step
    warmup_steps = float(warmup_steps)
    step = global_step + 1.0
    _start_step = warmup_steps
    lr = init_lr * warmup_steps**-1.0 * np.minimum(step, warmup_steps * anneal_rate**((step - _start_step) // anneal_interval))
    return lr

def step_weight_decay(init_lr, global_step, warmup_steps=50000):
    # exponential_decay with step
    warmup_steps = float(warmup_steps)
    step = global_step + 1.0
    _start_step = warmup_steps
    lr = init_lr * warmup_steps**-1.0 * np.minimum(step, warmup_steps)
    return lr

# def _pad(seq, max_len):
#     return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
    return x

def collate_fn(batch):
    """Create batch：手动将抽取出的样本堆叠起来的函数"""
    # (mel_data, embed_data, another_embed_data, spkid)
    # sort batch by frames
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    padded_len = max_input_len + hp.freq - max_input_len % hp.freq

    input_mel_batch = np.array([_pad_2d(x[0], padded_len) for x in batch], dtype=np.float32)
    input_mel_batch = torch.FloatTensor(input_mel_batch)

    #input_lengths_batch = torch.LongTensor(input_lengths)

    spkid_batch = np.array([x[1] for x in batch], dtype=np.int32)
    spkid_batch = torch.LongTensor(spkid_batch)

    input_wavLM_lengths = [len(x[2]) for x in batch]
    input_wavLM_lengths_batch = torch.LongTensor(input_wavLM_lengths)
    max_input_wavLM_len = np.max(input_wavLM_lengths)
    padded_wavLM_len = max_input_wavLM_len + hp.freq - max_input_wavLM_len % hp.freq

    input_wav2vec_batch = np.array([_pad_2d(x[2], padded_wavLM_len) for x in batch], dtype=np.float32)
    input_wav2vec_batch = torch.FloatTensor(input_wav2vec_batch)

    mono_file_list = [x[3] for x in batch]

    return input_mel_batch, input_wavLM_lengths_batch, spkid_batch, input_wav2vec_batch, mono_file_list

def save_checkpoint(model, optimizer, step, output_model_dir, data_parallel=False):
    checkpoint_path = os.path.join(output_model_dir, "model_{}.pt".format(step))
    torch.save({
        "state_dict": model.module.state_dict() if data_parallel else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, checkpoint_path)
    print("Saved model step:", checkpoint_path)

#log
def local_log_value(local_rank, *args, **kwargs):
    if local_rank == 0:
        log_value(*args, **kwargs)

def train(train_dataloader, output_model_dir, checkpoint_path, n_gpu=1, local_rank=0, CUDA_LAUNCH_BLOCKING=1):#分布式训练，进程的编号
#def train(train_dataloader, output_model_dir, checkpoint_path):
    use_cuda = hp.use_cuda and torch.cuda.is_available()
    assert use_cuda == True
    #device = torch.device("cuda", local_rank)
    #device = torch.device("cpu")
    print("line 99 {}".format(device))
    #print(torch.cuda.current_device())
    print("device: ", device)
    print("line 102 {}".format(device))

    model = model_wavLM_vq_ctc.Encoder()

    init_learning_rate = hp.learning_rate
    print("model init learning_rate is {0}".format(init_learning_rate))
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    model = model.to(device)

    step = 0
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #step = checkpoint["step"]
    print("train before {}".format(device))
    model.train()
    print("train after {}".format(device))
    l2_loss = seqloss.MaskedMSELoss()
    mse_loss = nn.MSELoss()
    l1_loss = seqloss.MaskedL1Loss()
    ce_loss = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss(blank=39, reduction='mean')

    mono_dict_file = '/ssdhome/wl/data/vctk_vad_wav/cmudict_phone'
    phone_dict = feat_dataset_copy.FeatDataset.read_mono_dict(mono_dict_file)
    #ipdb.set_trace()
    start_time = datetime.now().timestamp()
    for epoch in range(hp.epoch_num):
        for idx, (mel_data, input_lengths, spkid, wav2vec_data, mono_file_list) in enumerate(train_dataloader):
            batch_mono_list, batch_mono_lengths = feat_dataset_copy.FeatDataset.create_mono_bath(mono_file_list, phone_dict)
            batch_mono_lengths = np.array(batch_mono_lengths, dtype=np.int32)
            batch_mono_list = np.array(batch_mono_list, dtype=np.int32)
            
            batch_mono_list = torch.LongTensor(batch_mono_list)
            batch_mono_lengths = torch.LongTensor(batch_mono_lengths)
            ####wl 用wavLM帧数mask
            feature_length = wav2vec_data.shape[1]
            # feature_length = mel_data.shape[1]
            # print("feature_length {}".format(mel_data.shape))
            # print("wav2vec_data_length {}".format(wav2vec_data.shape))
            ctc_input_lengths = [feature_length] * hp.batchsize
            ctc_input_lengths = torch.LongTensor(ctc_input_lengths)

            mel_data, input_lengths, spkid, wav2vec_data, batch_mono_list, batch_mono_lengths, ctc_input_lengths = \
                mel_data.to(device), input_lengths.to(device), spkid.to(device), wav2vec_data.to(device), batch_mono_list.to(device), batch_mono_lengths.to(device), ctc_input_lengths.to(device)

            input_mask = seqloss.sequence_mask(input_lengths, max_len=feature_length, device=device).unsqueeze(-1)

            current_lr = init_learning_rate
            for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            seq_lengths = input_lengths # None for non-
            
            (mel_outputs, vq_loss, vq_perplexity, ctc_logits) = model( mel_data, wav2vec_data, seq_lengths)

            loss_mel = l2_loss(mel_outputs, wav2vec_data, mask=input_mask)
            loss = (loss_mel +  vq_loss * hp.loss_weight_vq)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_thresh)
            optimizer.step()

            local_log_value(local_rank, "learning_rate", current_lr, step)
            local_log_value(local_rank, "loss", loss, step)
            local_log_value(local_rank, "loss_mel", loss_mel, step)
            local_log_value(local_rank, "vq_loss", vq_loss, step)
            local_log_value(local_rank, "vq_perplexity", vq_perplexity, step)

            if step % hp.log_interval == 0 and step > 0:
                runtime = datetime.now().timestamp() - start_time
                print("lr is {0}".format(current_lr))
                log_string = "step {0}, loss {1:.8f}, loss_mel {2:.8f}, vq_loss {3:.8f}, ppl {4:.8f}, lr {5:.8f}, runtime {6:.3f}"\
                    .format(step, loss, loss_mel, vq_loss, vq_perplexity, current_lr, runtime)
                train_speed = hp.log_interval * hp.batchsize * n_gpu / runtime
                local_log_value(local_rank, 'train_speed', train_speed, step)
                print("local rank %d\t%s" % (local_rank, log_string))
                start_time = datetime.now().timestamp()

            if step % hp.checkpoint_interval == 0 and step > 0:
                if local_rank == 0:
                    save_checkpoint(model, optimizer, step, output_model_dir, n_gpu > 1)
            
            step = step + 1
            if step > hp.max_step:
                print('train done')
                sys.exit(1)


def main():
    global device
    device = "cuda:1"
    #print("device:{}".format(device))
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='wavLM+vq+ctc_bdl')
    parser.add_argument('--input', default='/ssdhome/wl/data/vctk_vad_wav/wavLM/wavLM_ctc_bdl.txt')
    parser.add_argument('--checkpoint_path', default='/ssdhome/wl/wl2021/vqvae_vq40_q64_base/outputs_lastest/output_wavLM+vq+ctc/wavLM+vq+ctc/model/model_150000.pt')
    parser.add_argument('--output_dir', default='./outputs_lastest/output_wavLM+vq+ctc')
    parser.add_argument('--gpus',default=1,help="gpu ids, eg. 0,1,2,3")
    parser.add_argument('--num_gpu', type=int, default=1, help="num gpus to use")
    parser.add_argument("--local_rank", type=int, default=0) #进程的编号
    args = parser.parse_args()

    run_name = args.name
    input_all_file = args.input
    checkpoint_path = args.checkpoint_path
    output_dir = os.path.join(args.output_dir, run_name)

    output_model_dir = '{0}/model'.format(output_dir)
    if args.local_rank == 0:
        os.makedirs(output_model_dir, exist_ok=True)
        tb_log_dir = 'tb_log'
        tensorboard_logger.configure("{0}/{1}".format(output_dir, tb_log_dir))

    train_dataset = feat_dataset_copy.FeatDataset(input_all_file, same_ref=True, training=True)
    sampler = feat_dataset_copy.RandomIdentitySamplerinfer(dataset=train_dataset, num_instances=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.batchsize, sampler=sampler,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    train(train_dataloader, output_model_dir, checkpoint_path, args.num_gpu, args.local_rank, CUDA_LAUNCH_BLOCKING=1)
    #(train_dataloader, output_model_dir, checkpoint_path)

if __name__ == '__main__':
    main()
