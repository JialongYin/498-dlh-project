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

from model import FactorGraphNN, BP_FGNN
from data import collate_wrapper

def get_args():
    pass

def run_evaluation(args, checkpoint, test_dict):
    pass

def main(args):
    tic = time.time()
    test_dict = joblib.load(args.pkl)
    print(args.run)
    # print("dataset len:", len([item for sublist in test_dict.values() for item in sublist]))
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    average = run_evaluation(args, checkpoint, test_dict)
    # print('[{:.2f}] Finish evaluation'.format(time.time() - tic))

if __name__ == '__main__':
    args = get_args()
    main(args)
