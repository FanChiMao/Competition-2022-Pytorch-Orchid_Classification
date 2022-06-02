import time
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.model_utils import build_model
from utils.train_utils import *
from utils.data_utils import *
import yaml

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml data
with open('train.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
PATH = opt['PATH']
OPT = opt['OPTIM']
MODEL = opt['MODEL']

## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Model
print('==> Build the model')
model = build_model(MODEL)
model.to(device)

## Setting path direction
csv_path = PATH['CSV_DIR']
images_folder = PATH['IMG_DIR']
model_save_path = PATH['RESULT_DIR']
log_dir = os.path.join(model_save_path, MODEL, 'log')
model_dir = os.path.join(model_save_path, MODEL, 'model')
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

## DataLoaders and transformation
print('==> Loading datasets')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(384),
        transforms.Normalize((0.5015, 0.4198, 0.3728), (0.2440, 0.2424, 0.2481)),
        # transforms.RandomCrop(384)
    ])
dataset = CustomDataset(csv_path, images_folder, transform=transform)
kfold = dataset_indexes(num_classes=Train['CLASS'], images_each_class=Train['IMG_EACH_CLASS'], split=10, fold=Train['FOLD'])

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Classifier:         {MODEL}
    Class number:       {Train['CLASS']}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['TRAIN_BATCH']}
    Learning rate:      {OPT['LR']}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
folds_val_best_accuracy = []
folds_val_best_best_precision = []
folds_val_best_best_recall = []
folds_val_best_best_f1 = []
best_epoch_acc = 0

for fold, (train_ids, test_ids) in enumerate(kfold):
    val_best_accuracy = 0
    val_best_precision = 0
    val_best_recall = 0
    val_best_f1 = 0

    train_subset = Subset(dataset, train_ids)
    test_subset = Subset(dataset, test_ids)
    trainloader = DataLoader(train_subset, batch_size=OPT['TRAIN_BATCH'], shuffle=True, pin_memory=True)
    testloader = DataLoader(test_subset, batch_size=OPT['VAL_BATCH'], pin_memory=True)

    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        epoch_start_time = time.time()
        train_epoch_loss = AverageMeter()
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(trainloader), 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.update(loss.item())

        # validation
        val_epoch_loss = AverageMeter()
        val_epoch_metric = MetricMeter(label=Train['CLASS'])

        model.eval()
        for i, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_epoch_loss.update(loss.item())
                predicts = torch.argmax(outputs, dim=1)
                val_cm = metric_(predicts, labels, num_classes=Train['CLASS'])
                val_epoch_metric.update(val_cm)

            # Val_loss : {val_epoch_loss.avg}

        if val_epoch_metric.o_acc >= val_best_accuracy:
            best_epoch_acc = epoch
            val_best_accuracy = val_epoch_metric.o_acc
            torch.save(model.state_dict(), os.path.join(model_dir, f'{MODEL}_fold{fold + 1}_best_acc.pth'))

        print("[fold %d epoch %d Acc: %.4f --- best_epoch %d Best_Acc %.4f]" % (
                fold+1, epoch, val_epoch_metric.o_acc, best_epoch_acc, val_best_accuracy))

        writer.add_scalar('val/accuracy', val_epoch_metric.o_acc, epoch)
        writer.add_scalar('val/loss', val_epoch_loss.avg, epoch)

        print("------------------------------------------------------------------")
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time() - epoch_start_time, train_epoch_loss.avg))
        print("------------------------------------------------------------------")
    folds_val_best_accuracy.append(val_best_accuracy)
    writer.add_scalar('train/loss', train_epoch_loss, epoch)
writer.close()

print(folds_val_best_accuracy)


