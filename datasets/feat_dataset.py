import os
import os.path as osp
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import random
import copy
from collections import defaultdict
import sys
sys.path.append('..')
import pdb
#from hparams import hparams as hp
#对语料库读取 Total speakers: 108, total samples: 43398
###wl:Total speakers: 109, total samples: 44237
class FeatDataset(Dataset):
    def __init__(self, datalst_path, same_ref=True, training=False):
        super(FeatDataset).__init__()
        self.same_ref = same_ref
        self.training = training
        self.sample_to_feat = {}
        self.spkid_to_samples = defaultdict(list)
        # maybe unuse, reserved
        self.spid_to_spkname = {}
        self.sample_to_spkid = {}
        self.parse_datalst(datalst_path)
        self.sample_ids = copy.copy(list(self.sample_to_feat.keys()))
        self.num_spkids = len(self.spkid_to_samples)
        if self.training:
            random.shuffle(self.sample_ids)
        print("Total speakers: %d, total samples: %d" % (len(self.spid_to_spkname), len(self.sample_ids)))

    def parse_datalst(self, datalst_path):
        with open(datalst_path, "r") as fp:
            lines = fp.readlines()
        
        for idx, line in enumerate(lines):
            speaker, speaker_id, sample_id, mel_path, mono_path = tuple(line.strip().split())
            speaker_id = int(speaker_id)
            #sample_id = idx
            assert sample_id not in self.sample_to_feat
            self.sample_to_feat[sample_id] = (mel_path, mono_path)
            self.spid_to_spkname[speaker_id] = speaker
            self.spkid_to_samples[speaker_id].append(sample_id)
            self.sample_to_spkid[sample_id] = speaker_id

    def read_mono_dict(dict_file):
        phone_dict = {}
        with open(dict_file) as f:
            lines = f.readlines()
            #for line in lines:
            for idx, line in enumerate(lines):
                #idx, phone = line.strip().split()
                phone = line.strip()
                phone_dict[phone] = idx
        print(phone_dict)
        return phone_dict

    def __len__(self):
        return len(self.sample_ids)

    def create_mono_bath(mono_file_list):
        batch_mono_lengths = []
        batch_mono_list = []
        for mono_path in mono_file_list:
            with open(mono_path) as f:
                mono_list = []
                lines = f.readlines()
                lines.decode("utf8","ignore")
                # for line in lines:
                #     phone_id = int(phone_dict[line.strip()])
                #     mono_list.append(phone_id)
            
            mono_len = len(mono_list)
            batch_mono_lengths.append(mono_len)
            batch_mono_list.extend(mono_list)
        return batch_mono_list, batch_mono_lengths, mono_list        

    #def __getitem__(self, idx):
    def __getitem__(self, sample_id):
        #sample_id = self.sample_ids[idx]
        mel_path, mono_path = self.sample_to_feat[sample_id]
        mel_data = np.load(mel_path)
        ####wl 加的
        wav2vec_data = np.load(mono_path)
        spkid = self.sample_to_spkid[sample_id]
        # print(mel_data.shape)
        # print(wav2vec_data.shape)
        #return mel_data, spkid, mono_path
        return mel_data, spkid, wav2vec_data


class FeatDatasetinfer(Dataset):
    def __init__(self, datalst_path, same_ref=True, training=False):
        super(FeatDatasetinfer).__init__()
        self.same_ref = same_ref
        self.training = training
        self.sample_to_feat = {}
        self.spkid_to_samples = defaultdict(list)
        # maybe unuse, reserved
        self.spid_to_spkname = {}
        self.parse_datalst(datalst_path)
        self.sample_ids = copy.copy(list(self.sample_to_feat.keys()))
        self.num_spkids = len(self.spkid_to_samples)
        if self.training:
            random.shuffle(self.sample_ids)
        print("Total speakers: %d, total samples: %d" % (len(self.spid_to_spkname), len(self.sample_ids)))

    def parse_datalst(self, datalst_path):
        with open(datalst_path, "r") as fp:
            lines = fp.readlines()

        for idx, line in enumerate(lines):
            speaker, speaker_id, sample_id, mel_path = tuple(line.strip().split())
            speaker_id = int(speaker_id)
            sample_id = idx
            assert sample_id not in self.sample_to_feat
            self.sample_to_feat[sample_id] = (mel_path)
            self.spid_to_spkname[speaker_id] = speaker
            self.spkid_to_samples[speaker_id].append(sample_id)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        mel_path = self.sample_to_feat[sample_id]
        mel_data = np.load(mel_path)

        return mel_data, spkid

class RandomIdentitySamplerinfer(Sampler):
    def __init__(self, dataset, num_instances=2, times=10):
        super(RandomIdentitySamplerinfer, self).__init__(dataset)
        self.dataset = dataset
        self.num_instances = num_instances
        self.times = times

    def __iter__(self):
        spkids = copy.copy(list(self.dataset.spkid_to_samples.keys()))
        #random.shuffle(spkids)
        for i in range(self.times):
            spkids.extend(spkids)

        sel_samples = []
        for spkid in spkids:
            sampleids = self.dataset.spkid_to_samples[spkid]
            replace = len(sampleids) < self.num_instances
            choice = np.random.choice(sampleids, size=self.num_instances, replace=replace)
            sel_samples.extend(choice)
        return iter(sel_samples)

    def __len__(self):
        return self.num_instances * self.dataset.num_spkids * self.times
#随机id采样
class RandomIdentitySampler(Sampler):
    def __init__(self, dataset, num_instances=1):
        super(RandomIdentitySampler, self).__init__(dataset)
        self.dataset = dataset
        self.num_instances = num_instances

    def __iter__(self):
        spkids = copy.copy(list(self.dataset.spkid_to_samples.keys()))
        random.shuffle(spkids)
        sel_samples = []
        for spkid in spkids:
            sampleids = self.dataset.spkid_to_samples[spkid]
            replace = len(sampleids) < self.num_instances
            choice = np.random.choice(sampleids, size=self.num_instances, replace=replace)
            sel_samples.extend(choice)

        return iter(sel_samples)

    def __len__(self):
        return self.num_instances * self.dataset.num_spkids


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """Create batch"""
    # (mel_data, embed_data, another_embed_data, spkid)

    # sort batch by frames
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    #batch1 = sorted(batch, key=lambda x: len(x[3]), reverse=True)

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    padded_len = max_input_len + 12 - max_input_len % 12

    #input_mel_batch = [_pad_2d(x[0], padded_len) for x in batch]
    input_mel_batch = np.array([_pad_2d(x[0], padded_len) for x in batch], dtype=np.float32)
    input_mel_batch = torch.FloatTensor(input_mel_batch)

    input_lengths_batch = torch.LongTensor(input_lengths)

    spkid_batch = np.array([x[1] for x in batch], dtype=np.int32)
    spkid_batch = torch.LongTensor(spkid_batch)
    
    #mono_file_list = [x[2] for x in batch]
    # input_lengths1 = [len(x[3]) for x in batch]
    # max_input_len1 = np.max(input_lengths1)
    # padded_len1 = max_input_len1 + 12 - max_input_len1 % 12
    # #max_len = np.max(wav2vec_data, key = lambda x: x[3].shape[1]).shape[1]
    # padded_len1 = max__len + 12 - max__len % 12
    # input_wav2vec_batch = np.array([_pad_2d(x[3], padded_len1) for x in batch], dtype=np.float32)
    # input_wav2vec_batch = torch.FloatTensor(input_wav2vec_batch)
    wav2vec_len = [len(x[2]) for x in batch]
    # print(wav2vec_len)
    #max_wav2vec_len = max(wav2vec_len)
    #print(max_wav2vec_len)
    #input_wav2vec_batch = _pad_2d([x[2].shape[1], max_wav2vec_len) for x in batch
    input_wav2vec_batch = np.array([_pad_2d(x[2], padded_len) for x in batch], dtype=np.float32)
    input_wav2vec_batch = torch.FloatTensor(input_wav2vec_batch)
    #return input_mel_batch, input_lengths_batch, spkid_batch, mono_file_list
    return input_mel_batch, input_lengths_batch, spkid_batch, input_wav2vec_batch

if __name__ == "__main__":
    import sys
    #data_lst = '/home/wjc505/vc_autoencoder/vctk_vad_wav/data_vctk_withmono.txt'
    data_lst = '/ssdhome/wl/data/vctk_vad_wav/data_vctk_wav2vec.txt'
    train_dataset = FeatDataset(data_lst, same_ref=True, training=True)
    sampler = RandomIdentitySampler(dataset=train_dataset, num_instances=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0, collate_fn=collate_fn, pin_memory=True, drop_last=True)

    #mono_dict_file = '/home/wjc505/vc_autoencoder/vctk_vad_wav/cmudict_phone'
    # mono_dict_file = '/ssdhome/wl/data/vctk_vad_wav/cmudict_phone'
    # phone_dict = FeatDataset.read_mono_dict(mono_dict_file)

    #for idx, (mel_data, input_lengths, spkid, mono_file_list) in enumerate(train_dataloader):
    ####wl
    for idx, (mel_data, input_lengths, spkid, wav2vec_data) in enumerate(train_dataloader):
        
        # batch_mono_list, batch_mono_lengths, mono_list = FeatDataset.create_mono_bath(mono_file_list) 
        # batch_mono_lengths = np.array(batch_mono_lengths, dtype=np.int32)
        # batch_mono_list = np.array(batch_mono_list, dtype=np.int32)
        
        print(mel_data.shape)
        # print(batch_mono_list)
        # print(batch_mono_lengths)
        print(input_lengths.shape)
        print(spkid.shape)
        print(wav2vec_data.shape)
        sys.exit(1)