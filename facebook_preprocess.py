import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import pandas as pd
import joblib
import networkx as nx
from PIL import Image
import os
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torchtext.data import Field, TabularDataset
import tarfile
import time

class Dataset(Dataset):
    def __init__(self, csv_file="MIMIC_CXR_dataset/"+"train.csv"):
        self.df = pd.read_csv(csv_file)
        # images file root
        # self.dataroot = 'files'
        self.dataroot = '/shared/rsaas/jialong2/physionet.org_depre/files/mimic-cxr-jpg/2.0.0/files'
        # untar images tar file to dataroot
        # self.dataroot = '/data/jialong2/files'
        # self.tar_path = '/shared/rsaas/jialong2/physionet.org_depre/files/mimic-cxr-jpg/2.0.0/files.tar.gz'
        # print("untar images file")
        # my_tar = tarfile.open(self.tar_path)
        # my_tar.extractall(self.dataroot) # specify which folder to extract to
        # my_tar.close()
        # print("complete untar images file")

        self.img_size = 128
        self.TEXT = Field(sequential=True, lower=True, include_lengths=True, batch_first=True) #, fix_length=200
        train_datafields = {'Text': ("text", self.TEXT)}
        self.train_dataset = TabularDataset(
           path=csv_file, # the file path
           format='csv',
           fields=train_datafields)
        self.TEXT.build_vocab(self.train_dataset)
        self.vocab_size = len(self.TEXT.vocab)

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, idx):
    #     rpt_raw, imgs_raw, cls = self.data[idx]
    #     rpt_word = ['<start>']+[word if word in self.word_to_ix else '<unk>' for word in rpt_raw]+['<end>']
    #     rpt_idx = torch.tensor([self.word_to_ix[word] for word in rpt_word])
    #     rpt_one_hot = torch.zeros(rpt_idx.size(0), self.vocab_size).scatter_(1, rpt_idx.unsqueeze(1), value=1)
    #     img = torch.tensor(imgs_raw[0]).unsqueeze(0)
    #     return rpt_one_hot, img, len(rpt_one_hot), torch.tensor(cls)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # get X ray image
        img_path = os.path.join(self.dataroot, 'p'+str(row['subject_id'])[:2],
                                    'p'+str(row['subject_id']), 's'+str(row['study_id']), str(row['dicom_id'])+'.jpg')
        img = Image.open(img_path)
        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)

        # # get report text
        # rpt_idx, rpt_len = self.TEXT.process([self.train_dataset[idx].text])
        # rpt_idx = rpt_idx.squeeze(0)
        #
        # # get cls
        # cls_series = row[3:-1]
        # cls = torch.tensor(cls_series.where(cls_series > 0, 0)).float()
        # return rpt_idx, img, rpt_len, cls
        return img

def main():
    global tic
    tic = time.time()
    print("dataset processing ...")
    train_dataset = Dataset()
    for i, img in enumerate(train_dataset):
        img = img.repeat(3, 1, 1)
        img = TF.to_pil_image(img)
        img.save("x_rays/"+"{}.jpg".format(i))
        if i % 1000 == 0:
            print(i)
    print('[{:.2f}] Finish training'.format(time.time() - tic))

if __name__ == '__main__':
    main()
