from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import argparse
from utils.model_utils import *
import os
from glob import glob
from utils.csv_utils import write_csv
from utils.data_utils import *
from utils.model_utils import *
import yaml

## Load yaml data
with open('predict_ensemble.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
MODEL = opt['MODEL']
PATH = opt['PATH']

inp_dir = PATH['INPUT_DIR']
out_dir = PATH['RESULT_DIR']
os.makedirs(out_dir, exist_ok=True)

## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Model
print('==> Build the models')

# predicts model
ensemble_models = build_ensemble_model(model=MODEL, pretrained=False)
models_predicts = ModelsPredicts(ensemble_models)
models_predicts = models_predicts  # half
models_predicts.to(device)

# ensemble model
print(len(ensemble_models))  # list
num_models = len(ensemble_models)
num_classes = Train['CLASS']

predicts_ensemble = PredictsEnsemble(num_models=num_models, num_classes=num_classes)
predicts_ensemble.to(device)

# load ensemble model
weights = PATH['ENS_PATH']
load_checkpoint(predicts_ensemble, weights)

datald_256 = set2loader_test(inp_dir, 256, OPT['TRAIN_BATCH'])
datald_320 = set2loader_test(inp_dir, 320, OPT['TRAIN_BATCH'])
datald_352 = set2loader_test(inp_dir, 352, OPT['TRAIN_BATCH'])
datald_384 = set2loader_test(inp_dir, 384, OPT['TRAIN_BATCH'])
datald_480 = set2loader_test(inp_dir, 480, OPT['TRAIN_BATCH'])

# [DATASET for ensemble model]
test_predicts = []

# output lists for result
predicts_list = []
image_names_list = []  # ok

print(f'Generating training data')
# [TESTING DATASET]
for i, ((inp_256, _), (inp_320, _), (inp_352, _), (inp_384, _), (inp_480, image_names)) in \
        enumerate(tqdm(zip(datald_256, datald_320, datald_352, datald_384, datald_480), total=len(datald_256))):
    # .to()
    inp_256 = inp_256.to(device)
    inp_320 = inp_320.to(device)
    inp_352 = inp_352.to(device)
    inp_384 = inp_384.to(device)
    inp_480 = inp_480.to(device)

    # predicts model
    # [batch, `models`, classes]
    outputs = models_predicts(inp_256, inp_320, inp_352, inp_384, inp_480)


    test_predicts.append(outputs)

    for image_name in image_names:
        image_names_list.append(image_name)


print('==> Start predicting')
result = []

predicts_ensemble.eval()
for i, (predicts) in enumerate(tqdm(zip(test_predicts), total=len(test_predicts))):
    with torch.no_grad():

        outputs = predicts_ensemble(predicts)
        predicts = torch.argmax(outputs, dim=1)

        preds = predicts.numpy().ravel()

        for pred in preds:
            predicts_list.append(pred)


result = [[x, y] for x, y in zip(image_names_list, predicts_list)]

write_csv(data=result, csv_path=out_dir)
print('finish !')
