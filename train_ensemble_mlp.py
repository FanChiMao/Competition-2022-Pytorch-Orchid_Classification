import yaml
import time
import random
from utils.model_utils import *
from utils.train_utils import *
from utils.data_utils import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
"""
Model used:

beit_base_patch16_384 [x2]      : accuracy 5x.x%    size: 384
volo_d2_384                     : accuracy 72.5%    size: 384
swin_base_patch4_window12_384   : accuracy 88.1%    size: 384
vit_base_patch16_384            : accuracy 80.7%    size: 384
convnext_base_384_in22ft1k      : accuracy 91.6%    size: 384
regnetz_e8                      : accuracy 87.3%    size: 320
ecaresnet50t                    : accuracy 84.1%    size: 320
tf_efficientnetv2_l_in21ft1k    : accuracy 85.9%    size: 480
dm_nfnet_f0                     : accuracy __  %    size: 256
"""

"""
https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-real.csv
interpolation mode : bicubic
"""
"""
model_224,
model_256,
model_320_1, model_320_2,
model_352,
model_384_1, model_384_2, model_384_3, model_384_4, model_384_5,
model_480

224 [resmlp_big_24_224_in22ft1k]
256 [dm_nfnet_f0]
320 [ecaresnet50t, regnetz_e8]
352 [ecaresnet269d]
384 [beit_base_patch16_384, swin_base_patch4_window12_384, vit_base_patch16_384, convnext_base_384_in22ft1k]
480 [tf_efficientnetv2_l_in21ft1k]
"""
## Load yaml data
with open('train.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
PATH = opt['PATH']
OPT = opt['OPTIM']
with open('predict_ensemble.yaml', 'r') as config:
    opt = yaml.safe_load(config)
MODEL = opt['MODEL']

## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Setting path direction
csv = PATH['CSV_DIR']
img = PATH['IMG_DIR']
model_save_path = PATH['RESULT_DIR']
log_dir = os.path.join(model_save_path, 'ENSEMBLE', 'log')
model_dir = os.path.join(model_save_path, 'ENSEMBLE', 'model')
mkdir(log_dir)
mkdir(model_dir)

## Log
writer = SummaryWriter(log_dir=log_dir)

## Model
from utils.model_utils import traditional_EnsembleModel
loaded_model_list = build_ensemble_model(model=MODEL, pretrained=False)

model = traditional_EnsembleModel(loaded_model_list)
model.to(device)

## Optimizer
start_epoch = 1
lr = float(OPT['LR_INIT'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

## Loss
criterion = nn.CrossEntropyLoss()

val_best_accuracy = 0
# train batch size
train_bs = 8
# test batch size
test_bs = 8

#
train_ids, test_ids = dataset_indexes(num_classes=219, images_each_class=80, split=10, fold=1)[0]  # only one in list

# different train / test dataloaders for different sizes
trainld_224, testld_224 = set2loader(csv, img, 224, train_ids, test_ids, train_bs, test_bs)
trainld_256, testld_256 = set2loader(csv, img, 256, train_ids, test_ids, train_bs, test_bs)
trainld_320, testld_320 = set2loader(csv, img, 320, train_ids, test_ids, train_bs, test_bs)
trainld_352, testld_352 = set2loader(csv, img, 352, train_ids, test_ids, train_bs, test_bs)
trainld_384, testld_384 = set2loader(csv, img, 384, train_ids, test_ids, train_bs, test_bs)
trainld_480, testld_480 = set2loader(csv, img, 480, train_ids, test_ids, train_bs, test_bs)

print('==> Start training ensemble model')
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    train_epoch_loss = AverageMeter()
    model.train()
    for i, ((inp_224, _), (inp_256, _), (inp_320, _), (inp_352, _), (inp_384, _), (inp_480, labels)) in \
            enumerate(tqdm(zip(trainld_224, trainld_256, trainld_320, trainld_352, trainld_384, trainld_480), total=len(trainld_224))):
        # print(i)
        inp_224 = inp_224.to(device)
        inp_256 = inp_256.to(device)
        inp_320 = inp_320.to(device)
        inp_352 = inp_352.to(device)
        inp_384 = inp_384.to(device)
        inp_480 = inp_480.to(device)
        labels = labels.to(device)

        outputs = model(inp_224, inp_256, inp_320, inp_352, inp_384, inp_480)
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
    for i, ((inp_224, _), (inp_256, _), (inp_320, _), (inp_352, _), (inp_384, _), (inp_480, labels)) in \
            enumerate(tqdm(zip(testld_224, testld_256, testld_320, testld_352, testld_384, testld_480))):
        # .to()
        inp_224 = inp_224.to(device)
        inp_256 = inp_256.to(device)
        inp_320 = inp_320.to(device)
        inp_352 = inp_352.to(device)
        inp_384 = inp_384.to(device)
        inp_480 = inp_480.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inp_224, inp_256, inp_320, inp_352, inp_384, inp_480)

            loss = criterion(outputs, labels)

            val_epoch_loss.update(loss.item())
            predicts = torch.argmax(outputs, dim=1)
            val_cm = metric_(predicts, labels, num_classes=Train['CLASS'])
            val_epoch_metric.update(val_cm)
        if val_epoch_metric.o_acc >= val_best_accuracy:
            best_epoch_acc = epoch
            val_best_accuracy = val_epoch_metric.o_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_acc.pth'))

        print("[epoch %d Acc: %.4f --- best_epoch %d Best_Acc %.4f]" % (
            epoch, val_epoch_metric.o_acc, best_epoch_acc, val_best_accuracy))
        writer.add_scalar('val/accuracy', val_epoch_metric.o_acc, epoch)
        writer.add_scalar('val/loss', val_epoch_loss.avg, epoch)
    print("------------------------------------------------------------------")
    print(
        "Epoch: {}\tTime: {:.4f}\tLoss: {:f}".format(epoch, time.time() - epoch_start_time, train_epoch_loss.avg))
    print("------------------------------------------------------------------")
writer.close()