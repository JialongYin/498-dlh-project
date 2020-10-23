import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict
import copy
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import CBN
from utils import *
from densenet import *


# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# DCGAN Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# DCGAN Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# class Generator(nn.Module):
#     def __init__(self, vocab_size, v_feat_size=512, hidden_size=2*512, word_emb_size=256, max_sen=10, max_word=20):
#         super(Generator, self).__init__()
#         # ? where to use spectral normalization -> "We also use spectral normalization for the layers in the generator and discriminator in the training process."
#         # embed noise & class
#         self.fc_cls = nn.Linear(14, 128)
#         self.fc_noise = nn.Linear(20, 20)
#         # image generator
#         self.G_img = G_img_Net(ResBlockUp, [1, 1, 1, 1, 1], 2)
#         # # report generator
#         # self.G_rpt = G_rpt_Net(vocab_size, v_feat_size, hidden_size, word_emb_size, max_sen, max_word)
#
#     def forward(self, noise, clss):
#         """Inputs: noise: batch x 120 x 1 x 1
#                     class: batch x class_num (14)
#            Return: images: batch x 1 x 128 x 128
#                     reports: batch x seq_len"""
#         # ? clss is not one-hot vector -> "class information y represented as one-hot vector."
#         # ? self.fc_noise # units not specified -> "The vectors zspl is passed through a linear layer to obtain zin"
#         # embed class from (*, 14) to (*, 128)
#         y_emd = self.fc_cls(clss)
#         # split (*, 120, 1, 1) noise to (*, 100, 1, 1) noise & (*, 20, 1, 1) z_spl->z_in
#         noise, z_spl = torch.split(noise, [100, 20], dim=1)
#         z_spl = z_spl.view(z_spl.size(0), -1)
#         z_in = self.fc_noise(z_spl)
#         # condition vector: concatenate from noise and class, batch x (128 + 20)
#         clss_emd = torch.cat((z_in, y_emd), dim=1)
#
#         # imgs: batch x 1 x 128 x 128
#         imgs = self.G_img(noise, clss_emd)
#         # # rpts: batch x seq_len
#         # rpts = self.G_rpt(imgs)
#         rpts = None
#         return imgs, rpts

# G_img_Net
class G_img_Net(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(G_img_Net, self).__init__()
        self.conv = nn.ConvTranspose2d(100, 64,
                  kernel_size=16, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.in_channels = 64

        self.layer0 = self.make_layer(block, 32, layers[0], 2)
        self.layer1 = self.make_layer(block, 16, layers[1], 2)
        self.attn_layer = SelfAttnBlock(16)
        self.layer2 = self.make_layer(block, 1, layers[2], 2)



    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, out_channels,
                          kernel_size=2, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, clss):
        # ? additional conv, bn, relu not aligned with "Architectural layout of EMIXER image generator"
        # ? Res-block-up channel not specified for each block -> "ci, co are input and output channels for the Res-block-up"
        # ? Res-block-up kernel size of Conv [4, 4, 1] & [3, 3, 1] mismatch -> "[3,3,1]"
        # ? Res-block-up kernel size of Shortcut [2, 2, 1] mismatch -> "[1,1,1]"
        # x: batch x 100 x 1 x 1
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # out: batch x 64 x 16 x 16
        out = self.layer0((out, clss))
        # out: batch x 32 x 32 x 32
        out = self.layer1((out, clss))
        # out: batch x 16 x 64 x 64
        out, attn = self.attn_layer(out)
        out = self.layer2((out, clss))
        # output: batch x 1 x 128 x 128
        return out

# G_rpt_Net
class G_rpt_Net(nn.Module):
    def __init__(self, vocab_size, v_feat_size=512, hidden_size=2*512, word_emb_size=256, max_sen=10, max_word=20):
        super(G_rpt_Net, self).__init__()
        # max num of sentences per report
        self.max_sen = max_sen
        # max num of words per sentence
        self.max_word = max_word

        # image encoder: pre-trained DenseNet model
        # ? not pretrianed yet, what different chest X-ray dataset -> "This CNN model is pre-trained on X-ray images I using a DenseNet model." "a different chest X-ray dataset"
        self.img_encoder = densenet121(num_classes=v_feat_size)
        self.sent_decoder = nn.LSTM(v_feat_size, hidden_size, 1, batch_first =True)
        # ? not use stop to decide sent num and use max_sent instead -> "get probability distribution ui over two states CONTINUE = 0, STOP = 1."
        self.stop_net = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        # ? self.topic_net # units not specified -> "a topic vector ti for ith sentence in the report"
        self.topic_net = nn.Sequential(
            nn.Linear(hidden_size, word_emb_size),
            nn.Linear(word_emb_size, word_emb_size),
            nn.Linear(word_emb_size, word_emb_size)
        )

        self.word_decoder = nn.LSTM(word_emb_size, hidden_size, 3, batch_first =True)
        self.embed = nn.Embedding(vocab_size, word_emb_size)
        self.word_pred_net = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, imgs):
        # imgs: batch x 1 x 128 x 128
        batch_size = imgs.size(0)
        # img_feat: batch x v_feat_size
        img_feat = self.img_encoder(imgs)


        # expand img_feat: batch x max_sen x v_feat_size
        img_feat = img_feat.unsqueeze(1).repeat(1,self.max_sen,1)
        # hidden: batch x max_sen x hidden_size
        hidden, (hn, cn) = self.sent_decoder(img_feat)
        # stops: batch x max_sen x 1
        stops = self.stop_net(hidden)
        # topics: batch x max_sen x word_emb_size
        topics = self.topic_net(hidden)


        # reshape topics: (batch * max_sen) x 1 x word_emb_size
        # topics for each sentence are independent also first input for word_rnn
        topics = topics.reshape(-1, topics.size(2)).unsqueeze(1)
        # {'<padding>':0, '<start>':1, '<end>':2, '<unk>':3}
        start = torch.ones(topics.size(0), dtype=torch.long)
        start_embed = self.embed(start).unsqueeze(1)
        # initial word_input of topics & start embedding: (batch * max_sen) x 2 x word_emb_size
        word_input = torch.cat((topics,start_embed), dim=1)

        rpts = []
        rpts_mask = torch.ones(batch_size, self.max_sen, self.max_word)
        # stops = (stops > 0.5).float()
        states = None
        for i in range(self.max_word):
            # hidden_word: (batch * max_sen) x 1 x hidden_size
            hidden_word, states = self.word_decoder(word_input, states)
            # word_token: (batch * max_sen) x 1
            word_token = self.word_pred_net(hidden_word[:, -1:, :]).argmax(dim=2)
            # word_input: (batch * max_sen) x 1 x word_emb_size
            word_input = self.embed(word_token)
            rpts.append(word_token)

        # rpts list to tensor: batch_size x (max_sen * max_word)
        rpts = torch.cat(rpts, dim=1).reshape(batch_size, -1)
        # rpts: batch_size x (start + max_sen * max_word)
        rpts = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), rpts), dim=1)
        return rpts

# class Discriminator(nn.Module):
#     def __init__(self, vocab_size, word_emb_size=512, hidden_size=2*512, v_feat_size=512):
#         super(Discriminator, self).__init__()
#         # # D_rpt
#         # self.embed = nn.Embedding(vocab_size, word_emb_size)
#         # self.D_rpt_rnn = nn.LSTM(word_emb_size, hidden_size, 1, batch_first =True)
#         # self.D_rpt_cls = nn.Linear(hidden_size, 2)
#         # D_img
#         self.D_img = D_img_Net(ResBlockDown, [1, 1, 1, 1, 1], 2)
#         # # D_joint
#         # self.D_joint = D_joint_Net(v_feat_size, vocab_size, word_emb_size, hidden_size)
#
#
#     def forward(self, input_imgs, input_rpts, input_clss):
#         # # embed input_rpts: batch x seq_len -> batch x seq_len x word_emb_size
#         # input_rpts_emb = self.embed(input_rpts)
#         # # h_n_rpts contains the hidden state for t = seq_len: (num_layers * num_directions, batch, hidden_size)
#         # _ , (h_n_rpts, _ ) = self.D_rpt_rnn(input_rpts_emb)
#         # # label_rpts: (num_layers * num_directions, batch, 2)
#         # label_rpts = self.D_rpt_cls(h_n_rpts)
#         # # label_rpts: (batch)
#         # label_rpts = F.softmax(label_rpts, dim=2)[:, :, 1].view(-1)
#
#
#         label_imgs = self.D_img(input_imgs, input_clss)
#         label_imgs = F.softmax(label_imgs, dim=1)[:, 1].view(-1)
#
#         # label_joint = self.D_joint(input_imgs, input_rpts)
#         # label_joint = F.softmax(label_joint, dim=1)[:, 1].view(-1)
#
#         label_rpts, label_joint = None, None
#         return label_imgs, label_rpts, label_joint

# D_img_Net
class D_img_Net(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(D_img_Net, self).__init__()
        # self.in_channels = 8
        # self.conv = conv3x3(1, 8)
        # self.bn = nn.BatchNorm2d(8)
        # self.relu = nn.ReLU(inplace=True)

        self.in_channels = 1
        self.layer0 = self.make_layer(block, 2, layers[0], 2)
        self.attn_layer = SelfAttnBlock(2)
        self.layer1 = self.make_layer(block, 4, layers[1], 2)
        self.layer2 = self.make_layer(block, 8, layers[2], 2)
        self.layer3 = self.make_layer(block, 16, layers[3], 2)
        self.layer4 = self.make_layer(block, 32, layers[4], 2)
        self.avg_pool = nn.AvgPool2d(2) #  nn.AvgPool2d需要添加参数ceil_mode=False，否则该模块无法导出为onnx格式
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128+14, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, clss):
        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.relu(out)

        # x: batch x 1 x 128 x 128
        out = self.layer0(x)
        out, attn = self.attn_layer(out)
        # out: batch x 2 x 64 x 64
        out = self.layer1(out)
        # out: batch x 4 x 32 x 32
        out = self.layer2(out)
        # out: batch x 8 x 16 x 16
        out = self.layer3(out)
        # out: batch x 16 x 8 x 8
        out = self.layer4(out)
        # out: batch x 32 x 4 x 4
        out = self.avg_pool(out)
        out = self.relu(out)
        # out: batch x 32 x 2 x 2
        out = out.view(out.size(0), -1)
        # clss: batch x 14
        out = torch.cat((out,clss), dim=1)
        out = self.fc(out)
        return out


# D_joint_Net
class D_joint_Net(nn.Module):
    def __init__(self, v_feat_size, vocab_size, word_emb_size, hidden_size):
        super(D_joint_Net, self).__init__()
        # img_feat
        self.img_encoder = densenet121(num_classes=v_feat_size)
        # rpt_feat
        self.embed = nn.Embedding(vocab_size, word_emb_size)
        self.D_rpt_rnn = nn.LSTM(word_emb_size, hidden_size, 1, batch_first =True)
        self.fc_feat = nn.Linear(hidden_size, hidden_size)
        # joint_feat classifier
        self.fc_cls = nn.Linear(hidden_size+v_feat_size, 2)

    def forward(self, input_imgs, input_rpts):
        batch_size = input_imgs.size(0)
        # img_feat: batch x v_feat_size
        img_feat = self.img_encoder(input_imgs)

        # embed input_rpts: batch x seq_len -> batch x seq_len x word_emb_size
        input_rpts = self.embed(input_rpts)
        # h_n_rpts contains the hidden state for t = seq_len: (num_layers * num_directions, batch, hidden_size)
        _ , (h_n_rpts, _ ) = self.D_rpt_rnn(input_rpts)
        # rpt_feat: batch x rpt_feat_size (hidden_size)
        rpt_feat = self.fc_feat(h_n_rpts).view(batch_size, -1)

        # joint_emb: batch x (v_feat_size + rpt_feat_size)
        joint_emb = torch.cat((img_feat, rpt_feat), dim=1)
        label_joint = self.fc_cls(joint_emb)
        return label_joint
