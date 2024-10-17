import albumentations as A
from glob import glob
from tqdm import tqdm
import math, random
import numpy as np
import timm
import os
from PIL import Image
import gc
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import wandb
import argparse
import json
from utils.logger import setup_logger
from utils.experiment_committer import git_push_process

parser = argparse.ArgumentParser(description="Git push for specific experiment")
parser.add_argument('--model_name', required=True, help='The name of the model')
parser.add_argument('--experiment_name', required=True, help='The name of the experiment')
args = parser.parse_args()

class cfg:
    seed = 123
    model_name = args.model_name
    experiment_name = args.experiment_name

git_push_process(model_name = cfg.model_name, experiment_name = cfg.experiment_name)

logger = setup_logger(model_name = cfg.model_name, experiment_name = cfg.experiment_name)
details_file = '/app/paper-implementations/vision/experiments/details.json'
with open(details_file, 'r') as f:
    details = json.load(f)

summary = details[cfg.model_name][cfg.experiment_name].get('summary', 'No summary available.')
logger.info(f'summary : {summary}')

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
set_seed(cfg.seed)
# something to written