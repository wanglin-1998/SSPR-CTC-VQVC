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
###wl:Total speakers: 109, total samples: 44237   没有315的说话人
#用的话改两个地方 16、115行
split_speaker = "/ssdhome/wl/SRD-VC-master/My_model/assets/slt.txt"
#split_speaker = "/ssdhome/wl/SRD-VC-master/My_model/assets/split_speaker.txt"
with open(split_speaker, 'r', encoding='utf-8') as f:
    c = f.read().strip().split('\n')[1].strip().split(' ')
    #print(len(c))       # 100

def speaker2index(string):
    id = c.index(string)
    return id

class FeatDataset(Dataset):
    def __init__(self, datalst_path, same_ref=True, training=False):
        super(FeatDataset).__init__()
        self.same_ref = same_ref
        self.training = training
        self.sample_to_feat = {}
        self.spkid_to_samples = defaultdict(list)
        # maybe unuse, reserved
        self.spid_to_spkname = {}  ###{0: 'p225', 1: 'p226', 2: 'p227',
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
            speaker, speaker_id, sample_id, mel_path, pre_path, mono_path = tuple(line.strip().split())  #speaker:'p225' speaker_id:'0'
            speaker_id = int(speaker_id)
            #sample_id = idx
            assert sample_id not in self.sample_to_feat

            # utterances = []
            # spkid = np.zeros((109,), dtype=np.float32)
            # spkid[speaker2index(speaker)] = 1.0
            # utterances.append(spkid)
            # # create file list
            # #for fileName in sorted(sample_id):
            # utterances.append(os.path.join(speaker,sample_id))
            # speakers.append(utterances)
            #wl 改动
            self.sample_to_feat[sample_id] = (mel_path, pre_path, mono_path)
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

    def create_mono_bath(mono_file_list, phone_dict):
        batch_mono_lengths = []
        batch_mono_list = []
        for mono_path in mono_file_list:
            with open(mono_path) as f:
                mono_list = []
                lines = f.readlines()
                #lines.decode("utf8","ignore")
                for line in lines:
                    phone_id = int(phone_dict[line.strip()])
                    mono_list.append(phone_id)
            mono_len = len(mono_list)
            batch_mono_lengths.append(mono_len)
            batch_mono_list.extend(mono_list)
        return batch_mono_list, batch_mono_lengths     

    def __getitem__(self, sample_id):
        #sample_id = self.sample_ids[idx]
        mel_path, pre_path, mono_path = self.sample_to_feat[sample_id]
        mel_data = np.load(mel_path)
        ####wl 加的
        wav2vec_data = np.load(pre_path)
        #mono_path = np.load(mono_path)
        spkid = self.sample_to_spkid[sample_id]
        

        spkname = self.spid_to_spkname [spkid]
        # print("mel shape {}".format(mel_data.shape))
        # print("prewithVCTK_wavLM shape {}".format(wav2vec_data.shape))
        # print("960h_wavLM shape {}".format(mono_path.shape))

        # utterances = []
        #spkid_grl = np.zeros((108,), dtype=np.float32)
        #### VCTK 数据集的
        #spkid_grl = np.zeros((108,))
        #### slt 数据集的
        spkid_grl = np.zeros((1,))
        spkid_grl[speaker2index(spkname)] = 1.0
        # utterances.append(spkid)
        # # create file list
        # #for fileName in sorted(sample_id):
        # utterances.append(os.path.join(speaker,sample_id))
        # speakers.append(utterances)

        # print(mel_data.shape)
        # print(wav2vec_data.shape)
        #return mel_data, spkid, mono_path
        return mel_data, spkid, wav2vec_data, mono_path, spkid_grl

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



def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)], mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """Create batch"""
    # (mel_data, embed_data, another_embed_data, spkid)
    # sort batch by frames
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    input_lengths = [len(x[0]) for x in batch] #[289, 213, 187, 179, 123, 119, 107, 105]
    max_input_len = np.max(input_lengths)  #289
    #padded_len = max_input_len + hp.freq - max_input_len % hp.freq
    padded_len = max_input_len + 12 - max_input_len % 12  #300

    input_mel_batch = np.array([_pad_2d(x[0], padded_len) for x in batch], dtype=np.float32)
    input_mel_batch = torch.FloatTensor(input_mel_batch)

    input_lengths_batch = torch.LongTensor(input_lengths)  #tensor([289, 213, 187, 179, 123, 119, 107, 105])

    spkid_batch = np.array([x[1] for x in batch], dtype=np.int32)
    spkid_batch = torch.LongTensor(spkid_batch)

    input_wavLM_lengths = [len(x[2]) for x in batch]
    input_wavLM_lengths_batch = torch.LongTensor(input_wavLM_lengths)
    max_input_wavLM_len = np.max(input_wavLM_lengths)
    padded_wavLM_len = max_input_wavLM_len + 12 - max_input_wavLM_len % 12
    input_wav2vec_batch = np.array([_pad_2d(x[2], padded_wavLM_len) for x in batch], dtype=np.float32)
    input_wav2vec_batch = torch.FloatTensor(input_wav2vec_batch)

    
    mono_file_list = [x[3] for x in batch]

    # input_mono_file_list = np.array([_pad_2d(x[3], padded_len) for x in batch], dtype=np.float32)
    # mono_file_list = torch.FloatTensor(mono_file_list)
    spkid_grl_batch = np.array([x[4] for x in batch], dtype=np.float32)
    spkid_grl_batch = torch.from_numpy(spkid_grl_batch)
    #spkid_grl_batch = torch.LongTensor(spkid_grl_batch)

    return input_mel_batch, input_wavLM_lengths_batch, spkid_batch, input_wav2vec_batch, mono_file_list, spkid_grl_batch

if __name__ == "__main__":
    import sys
    #data_lst = '/ssdhome/wl/data/vctk_vad_wav/wavLM/wavLM_ctc.txt'
    data_lst = '/ssdhome/wl/data/vctk_vad_wav/wavLM/wavLM_ctc_slt.txt'
    train_dataset = FeatDataset(data_lst, same_ref=True, training=True)
    sampler = RandomIdentitySamplerinfer(dataset=train_dataset, num_instances=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0, collate_fn=collate_fn, pin_memory=True, drop_last=True)

    # mono_dict_file = '/ssdhome/wl/data/vctk_vad_wav/cmudict_phone'
    # phone_dict = FeatDataset.read_mono_dict(mono_dict_file)

    #for idx, (mel_data, input_lengths, spkid, mono_file_list) in enumerate(train_dataloader):
    ####wl
    for idx, (mel_data, input_lengths, spkid, wav2vec_data, mono_file_list, spkid_grl_batch) in enumerate(train_dataloader):
        
        # batch_mono_list, batch_mono_lengths, mono_list = FeatDataset.create_mono_bath(mono_file_list) 
        # batch_mono_lengths = np.array(batch_mono_lengths, dtype=np.int32)
        # batch_mono_list = np.array(batch_mono_list, dtype=np.int32)
        
        print(mel_data.shape)  ##torch.Size([8, 792, 80])
        # print(batch_mono_list)
        # print(batch_mono_lengths)
        # print(input_lengths.shape)
        # print(spkid.shape) ##tensor([67, 38, 74, 34, 40, 68, 32, 46])
        print(wav2vec_data.shape)  ##torch.Size([8, 792, 1024])
        #print(mono_file_list.shape)
        print(spkid_grl_batch.shape)
        sys.exit(1)