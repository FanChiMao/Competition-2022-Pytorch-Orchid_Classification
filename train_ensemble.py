import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from glob import glob
import argparse
from utils.model_utils import *
from tensorboardX import SummaryWriter
import yaml
# from natsort import natsorted # ?
# from utils.csv_utils import write_csv # ?
from utils.model_utils import *
from utils.train_utils import *
from utils.data_utils import *
from utils.model_utils import build_model
import pandas as pd
import numpy as np


## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml data
with open('predict_ensemble.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
MODEL = opt['MODEL']
PATH = opt['PATH']

inp_dir = PATH['INPUT_DIR']  # no use yet
out_dir = PATH['RESULT_DIR']
os.makedirs(out_dir, exist_ok=True)
## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Model
print('==> Build the model')
ensemble_models = build_ensemble_model(model=MODEL, pretrained=False)
model = EnsembleModel(ensemble_models)
model.to(device)

## Setting path direction
csv_path = PATH['CSV_DIR']
images_folder = PATH['IMG_DIR']
model_save_path = PATH['RESULT_DIR']
log_dir = os.path.join(model_save_path, 'ensemble', 'log')
model_dir = os.path.join(model_save_path, 'ensemble', 'model')
mkdir(log_dir)
mkdir(model_dir)

## Log
writer = SummaryWriter(log_dir=log_dir)

## Optimizer
start_epoch = 1
lr = float(OPT['LR'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

## Loss
criterion = nn.CrossEntropyLoss()

# train / test index
train_ids, test_ids = dataset_indexes(num_classes=Train['CLASS'], images_each_class=Train['IMG_EACH_CLASS'], split=10,
                                      fold=Train['FOLD'])[0]  # only one in list

# different train / test dataloaders for different sizes
# trainld_224, testld_224 = set2loader(csv_path, images_folder, 224, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])
trainld_256, testld_256 = set2loader(csv_path, images_folder, 256, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])
trainld_320, testld_320 = set2loader(csv_path, images_folder, 320, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])
trainld_352, testld_352 = set2loader(csv_path, images_folder, 352, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])
trainld_384, testld_384 = set2loader(csv_path, images_folder, 384, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])
trainld_480, testld_480 = set2loader(csv_path, images_folder, 480, train_ids, test_ids, OPT['TRAIN_BATCH'], OPT['VAL_BATCH'])

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Classifier:             {f'Ensemble Model'}
    Class number:           {Train['CLASS']}
    Start/End epochs:       {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:            {OPT['TRAIN_BATCH']}
    Learning rate:          {OPT['LR']}''')
print('------------------------------------------------------------------')

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    train_epoch_loss = AverageMeter()
    model.train()
    for i, ((inp_256, _), (inp_320, _), (inp_352, _), (inp_384, _), (inp_480, labels)) in \
            enumerate(tqdm(zip(trainld_256, trainld_320, trainld_352, trainld_384, trainld_480), total=len(trainld_256))):

        # .to()
        inp_256 = inp_256.to(device)
        inp_320 = inp_320.to(device)
        inp_352 = inp_352.to(device)
        inp_384 = inp_384.to(device)
        inp_480 = inp_480.to(device)
        labels = labels.to(device)

        outputs = model(inp_256, inp_320, inp_352, inp_384, inp_480)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss.update(loss.item())

        writer.add_scalar('train/loss', train_epoch_loss.avg, epoch)

    # validation
    val_epoch_loss = AverageMeter()
    val_epoch_metric = MetricMeter(label=Train['CLASS'])

    model.eval()
    for i, ((inp_256, _), (inp_320, _), (inp_352, _), (inp_384, _), (inp_480, labels)) in \
            enumerate(tqdm(zip(testld_256, testld_320, testld_352, testld_384, testld_480), total=len(testld_256))):
        # .to()
        inp_256 = inp_256.to(device)
        inp_320 = inp_320.to(device)
        inp_352 = inp_352.to(device)
        inp_384 = inp_384.to(device)
        inp_480 = inp_480.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            outputs = model(inp_256, inp_320, inp_352, inp_384, inp_480)

            loss = criterion(outputs, labels)

            val_epoch_loss.update(loss.item())
            predicts = torch.argmax(outputs, dim=1)
            val_cm = metric_(predicts, labels, num_classes=Train['CLASS'])
            val_epoch_metric.update(val_cm)

    writer.add_scalar('val/accuracy', val_epoch_metric.o_acc, epoch)
    writer.add_scalar('val/loss', val_epoch_loss.avg, epoch)

    if val_epoch_metric.o_acc >= val_best_accuracy:
        best_epoch_acc = epoch
        val_best_accuracy = val_epoch_metric.o_acc
        torch.save(model.state_dict(), os.path.join(model_dir, f'ensemble_best_acc.pth'))

    print("[ epoch %d Acc: %.4f --- best_epoch %d Best_Acc %.4f]" % (epoch, val_epoch_metric.o_acc, best_epoch_acc, val_best_accuracy))

writer.close()