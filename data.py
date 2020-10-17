import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import pandas as pd
import joblib
import networkx as nx
from PIL import Image

class Dataset(Dataset):
    def __init__(self, pkl_file):
        self.data, self.word_bag = joblib.load(pkl_file)
        self.THRESHOLD = 1 #6
        "Filter words with occurance less than THRESHOLD"
        self.word_bag_filtered = {k:v for k,v in self.word_bag.items() if v >= self.THRESHOLD}
        "Build word index mappings"
        fun_dict = {'<padding>':0, '<start>':1, '<end>':2, '<unk>':3}
        self.word_to_ix = {k:i for i,k in enumerate(self.word_bag_filtered.keys(),start=len(fun_dict))}
        self.word_to_ix.update(fun_dict)
        self.ix_to_word = dict([reversed(i) for i in self.word_to_ix.items()])
        self.vocab_size = len(self.word_to_ix)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     rpt_raw, imgs_raw, cls = self.data[idx]
    #     rpt_word = ['<start>']+[word if word in self.word_to_ix else '<unk>' for word in rpt_raw]+['<end>']
    #     rpt_idx = torch.tensor([self.word_to_ix[word] for word in rpt_word])
    #     rpt_one_hot = torch.zeros(rpt_idx.size(0), self.vocab_size).scatter_(1, rpt_idx.unsqueeze(1), value=1)
    #     img = torch.tensor(imgs_raw[0]).unsqueeze(0)
    #     return rpt_one_hot, img, len(rpt_one_hot), torch.tensor(cls)

    def __getitem__(self, idx):
        rpt_raw, imgs_raw, cls = self.data[idx]
        rpt_word = ['<start>']+[word if word in self.word_to_ix else '<unk>' for word in rpt_raw]+['<end>']
        rpt_idx = torch.tensor([self.word_to_ix[word] for word in rpt_word])
        # rpt_one_hot = torch.zeros(rpt_idx.size(0), self.vocab_size).scatter_(1, rpt_idx.unsqueeze(1), value=1)
        img = torch.tensor(imgs_raw[0]).unsqueeze(0)
        # return rpt_one_hot, img, len(rpt_one_hot), torch.tensor(cls)
        return rpt_idx, img, len(rpt_idx), torch.tensor(cls)

# def collate_wrapper(batch):
#     rpts, imgs, rpt_lens, clss = list(zip(*batch))
#     rpts_batch = pad_sequence(rpts, batch_first=True)
#     imgs_batch = torch.stack(imgs)
#     labels_batch = torch.stack(clss)
#     rpt_lens, sort_index = torch.sort(torch.tensor(rpt_lens), descending=True)
#
#     rpts_batch = pack_padded_sequence(rpts_batch[sort_index], rpt_lens, batch_first=True)
#     imgs_batch = imgs_batch[sort_index]
#     labels_batch = labels_batch[sort_index]
#     return rpts_batch.float(), imgs_batch.float(), labels_batch.float()

def collate_wrapper(batch):
    # rpts list: batch x variale rpt length
    rpts, imgs, rpt_lens, clss = list(zip(*batch))
    # rpts_batch: batch x max rpt length
    rpts_batch = pad_sequence(rpts, batch_first=True)

    imgs_batch = torch.stack(imgs)
    labels_batch = torch.stack(clss)
    rpt_lens, sort_index = torch.sort(torch.tensor(rpt_lens), descending=True)

    # rpts_batch = pack_padded_sequence(rpts_batch[sort_index], rpt_lens, batch_first=True)
    imgs_batch = imgs_batch[sort_index]
    labels_batch = labels_batch[sort_index]
    return rpts_batch, imgs_batch.float(), labels_batch.float()
