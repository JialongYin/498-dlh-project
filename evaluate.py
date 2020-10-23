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

from data import Dataset, collate_wrapper
from model import Generator, Discriminator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    args.result_path = "results/"
    args.checkpoint_dir = "checkpoint_emixer/"
    args.checkpoint = "checkpoint"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.run) == 0:
        run_count = len([dir for dir in os.listdir(args.checkpoint_dir) if dir[0:3] == "run"])
        args.run = 'run{}'.format(run_count-1)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run)
    args.result_path = os.path.join(args.result_path, args.run)+'/'
    os.makedirs(args.result_path, exist_ok=True)
    return args

def run_evaluation(args, checkpoint):
    device = args.device
    # Create the generator
    netG = Generator(vocab_size=checkpoint['vocab_size']).to(device)
    # Create the Discriminator
    # netD = Discriminator(vocab_size=checkpoint['vocab_size']).to(device)

    netG.load_state_dict(checkpoint['G model_state_dict'])
    # netD.load_state_dict(checkpoint['D model_state_dict']).to(args.device)
    netG.eval()


    # img_list = []
    # fixed_noise = torch.randn(16, 120, 1, 1, device=device)
    b_size = 4
    nz = 120
    fixed_noise = torch.zeros(b_size, nz, 1, 1, device=device)
    for i in range(b_size):
        fixed_noise[i][i][0][0] = 1
    fixed_clss = torch.zeros((b_size, 14))
    fixed_clss[:, [8]] = 1
    fake_imgs, fake_rpts = netG(fixed_noise, fixed_clss)
    fake = fake_imgs.detach().cpu()
    for i in range(len(fake)):
        save_image(fake[i], args.result_path+'img'+str(i)+'.png')
    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # plt.show()

def main(args):
    tic = time.time()
    print(args.run)
    # print("dataset len:", len([item for sublist in test_dict.values() for item in sublist]))
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    average = run_evaluation(args, checkpoint)
    # print('[{:.2f}] Finish evaluation'.format(time.time() - tic))

if __name__ == '__main__':
    args = get_args()
    main(args)
