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

nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 120

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
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.fc_cls = nn.Linear(14, 128)
        self.fc_noise = nn.Linear(20, 20)
        self.G_img = G_img_Net(ResBlockUp, [1, 1, 1, 1, 1], 2)

    def forward(self, noise, clss):
        """Inputs: (a) noise vector z ∈ R^120 (b) class information y"""
        # print("noise", noise.size())
        # print(self.main)
        y_emd = self.fc_cls(clss)
        noise, z_spl = torch.split(noise, [100, 20], dim=1)
        z_spl = z_spl.view(z_spl.size(0), -1)
        z_in = self.fc_noise(z_spl)
        clss_emd = torch.cat((z_in, y_emd), dim=1)
        imgs = self.G_img(noise, clss_emd)
        # imgs = self.main(noise)
        # print("noise imgs", imgs.size())

        rpts = None
        return imgs, rpts

# G_img_Net
class G_img_Net(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(G_img_Net, self).__init__()
        self.in_channels = 64
        self.conv = nn.ConvTranspose2d(100, 64,
                  kernel_size=16, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

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
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, clss):
        # print("x", x.size())
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # print("conv", out.size())
        out = self.layer0((out, clss))
        # print("layer0", out.size())
        out = self.layer1((out, clss))
        # print("layer1", out.size())
        out, attn = self.attn_layer(out)
        out = self.layer2((out, clss))
        # print("layer2", out.size())
        return out

# Residual block
class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlockUp, self).__init__()
        self.condBN1 = CBN(128+20, 128+20, in_channels)
        self.conv1 = deconv4x4(in_channels, out_channels, stride)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.condBN2 = CBN(128+20, 128+20, out_channels)
        self.conv2 = deconv3x3(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x_clss_tuple):
        x, clss = x_clss_tuple
        # print("ResBlockUp", x.size(), clss.size())
        # print("x", x.size())
        residual = x
        out = self.condBN1(x, clss)
        out = self.relu(out)
        # print("conv1", self.conv1)
        out = self.conv1(out)
        # print("conv1", out.size())
        # out = self.bn1(out)
        out = self.condBN2(out, clss)
        out = self.relu(out)
        # print("conv2", self.conv2)
        out = self.conv2(out)
        # print("conv2", out.size())
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            # print("residual x", x.size())
            # print(self.downsample)
            # print("residual", residual.size())
        out += residual
        # out = self.relu(out)
        return out

# 3x3 convolution H->H
def deconv3x3(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# 4x4 convolution H->2H
def deconv4x4(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                     stride=stride, padding=1, bias=False)



class Discriminator(nn.Module):
    def __init__(self, ngpu, vocab_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.D_rpt_rnn = nn.LSTM(vocab_size, 2*vocab_size, 1, batch_first =True)
        self.D_rpt_cls = nn.Linear(2*vocab_size, 2)
        self.D_img = D_img_Net(ResBlockDown, [1, 1, 1, 1, 1], 2)


    def forward(self, input_imgs, input_rpts, input_clss):
        # _ , (h_n_rpts, _ ) = self.D_rpt_rnn(input_rpts)
        # label_rpts = self.D_rpt_cls(h_n_rpts)
        # label_rpts = F.softmax(label_rpts, dim=2)[:, :, 1].view(-1)

        label_imgs = self.D_img(input_imgs, input_clss)
        label_imgs = F.softmax(label_imgs, dim=1)[:, 1].view(-1)

        label_rpts = None
        label_joint = None
        return label_imgs, label_rpts, label_joint

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
        # print("initial", x.size())
        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.relu(out)
        out = self.layer0(x)
        out, attn = self.attn_layer(out)
        # print("layer0", out.size())
        out = self.layer1(out)
        # print("layer1", out.size())
        out = self.layer2(out)
        # print("layer2", out.size())
        out = self.layer3(out)
        # print("layer3", out.size())
        out = self.layer4(out)
        # print("layer4", out.size())
        out = self.avg_pool(out)
        out = self.relu(out)
        # print("output", out.size())
        out = out.view(out.size(0), -1)
        out = torch.cat((out,clss), dim=1)
        out = self.fc(out)
        return out

# Residual block
class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlockDown, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # out = self.relu(out)
        return out

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Self attention block
class SelfAttnBlock(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttnBlock,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = max(in_dim//8,1) , kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = max(in_dim//8,1) , kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention
