import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joblib
from scipy.stats import entropy
import matplotlib.pyplot as plt
import statistics
import math
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
from fid_score import FidScore
import torchvision.transforms.functional as TF

from data import Dataset, collate_wrapper
from model import Generator, Discriminator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    args.result_path = "results/"
    args.checkpoint_dir = "checkpoint_emixer/"
    args.checkpoint = "checkpoint"
    args.dataset = "MIMIC_CXR_dataset/"
    args.real_img_dir = "real_imgs/"
    args.fake_img_dir = "fake_imgs/"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.run) == 0:
        run_count = len([dir for dir in os.listdir(args.checkpoint_dir) if dir[0:3] == "run"])
        args.run = 'run{}'.format(run_count-1)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run)
    args.result_path = os.path.join(args.result_path, args.run)+'/'
    args.fake_img_dir = os.path.join(args.result_path, args.fake_img_dir)+'/'
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.real_img_dir, exist_ok=True)
    os.makedirs(args.fake_img_dir, exist_ok=True)
    return args

def run_evaluation(args, checkpoint, test_dataset):
    dataset_folder_generation(test_dataset)
    device = args.device
    # Create the generator
    # netG = Generator(vocab_size=checkpoint['vocab_size'], nz=checkpoint['args']['nz'], ngf=checkpoint['args']['ngf']).to(device)
    netG = Generator(vocab_size=checkpoint['vocab_size']).to(device)
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['G model_state_dict'])
    netG.eval()

    b_size = 1
    nz = checkpoint['args']['nz']
    for i, (_ , _ , _ , cls) in enumerate(test_dataset):
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        cls = cls.unsqueeze(0)
        fake_imgs, fake_rpts = netG(noise, cls)
        fake = fake_imgs.cpu().squeeze(0)
        fake = fake.repeat(3, 1, 1)
        img = TF.to_pil_image(fake)
        img.save(args.fake_img_dir+"{}.jpg".format(i))

    fid = FidScore([args.real_img_dir, args.fake_img_dir], device, len(test_dataset))
    score = fid.calculate_fid_score()
    print("fid scores:{}".format(score))
    args_file = open(args.result_path+'/fid_{}.txt'.format(score), "w")
    args_file.write("fid scores:{} ".format(score))
    args_file.close()

def dataset_folder_generation(test_dataset):
    for i, (_ , img, _ , _) in enumerate(test_dataset):
        img = img.repeat(3, 1, 1)
        img = TF.to_pil_image(img)
        img.save(args.real_img_dir+"{}.jpg".format(i))


def main(args):
    tic = time.time()
    print(args.run)
    test_dataset = Dataset(args.dataset+"test.csv") # "train.csv" "test.csv" validate.csv
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    average = run_evaluation(args, checkpoint, test_dataset)
    print('[{:.2f}] Finish evaluation {}'.format(time.time() - tic, args.run))

if __name__ == '__main__':
    args = get_args()
    main(args)
