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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
import json
from torch import nn, optim
from torchvision import datasets, transforms
from utils.logger import setup_logger
from utils.experiment_committer import git_push_process
from models.alexnet import AlexNet
import wandb 

parser = argparse.ArgumentParser(description="Git push for specific experiment")
parser.add_argument('--model_name', required=True, help='The name of the model')
parser.add_argument('--experiment_name', required=True, help='The name of the experiment')
args = parser.parse_args()

class cfg:
    seed = 123
    model_name = args.model_name
    experiment_name = args.experiment_name
    num_epochs = 10

# python train.py --model_name alexnet --experiment_name init
# wandb.init(project=cfg.model_name, name=cfg.experiment_name)
# git_push_process(model_name = cfg.model_name, experiment_name = cfg.experiment_name)

logger = setup_logger(model_name = cfg.model_name, experiment_name = cfg.experiment_name)
details_file = '/app/paper_implementations/vision/experiments/details.json'
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
set_seed(cfg.seed)


transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
train_ds = datasets.ImageFolder(root='/app/data/paper_implementations/vision/imagenet/train', transform = transform)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
valid_ds = datasets.ImageFolder(root='/app/data/paper_implementations/vision/imagenet/valid', transform = transform)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=1000, dropout=0.5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1, cfg.num_epochs+1):
    model.train()
    total_loss = 0.0

    # training phase
    with tqdm(train_dl, dynamic_ncols=True, mininterval=5.0, leave=True) as pbar:
        optimizer.zero_grad()

        for idx, batch in enumerate(pbar):
            images, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    train_loss = total_loss/len(train_dl)
    if cfg.experiment_name != '':
        wandb.log({"epoch": epoch, "train_loss": train_loss})

    # evaluation phase
    model.eval()  
    correct = 0
    total = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_dl:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_val_loss += loss.item()

    val_loss = total_val_loss / len(valid_dl)
    accuracy = 100 * correct / total

    if cfg.experiment_name != '':
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_accuracy": accuracy})