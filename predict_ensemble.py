from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import argparse
from utils.model_utils import *
import os
from natsort import natsorted
from glob import glob
from utils.csv_utils import write_csv
import yaml

if __name__ == '__main__':
    ## Load yaml data
    with open('predict_ensemble.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    MODEL = opt['MODEL']
    PATH = opt['PATH']

    inp_dir = PATH['INPUT_DIR']
    out_dir = PATH['RESULT_DIR']
    os.makedirs(out_dir, exist_ok=True)
    files = natsorted(glob(os.path.join(inp_dir, '*.JPG')) + glob(os.path.join(inp_dir, '*.PNG')))
    if len(files) == 0:
        raise Exception(f"No testing images in {inp_dir}")
    ensemble_models = build_ensemble_model(model=MODEL, pretrained=False)

    print('==> Start predicting')

    result = []
    for i, file_ in enumerate(tqdm(files)): # each image
        predict_result = []
        image_name = os.path.split(file_)[-1]
        img = Image.open(file_).convert('RGB')
        for index, (model, transformation, size) in enumerate(ensemble_models): # each ensemble model
            input_ = transformation(img).unsqueeze(0).cuda()
            with torch.no_grad():
                predict_vector = model(input_) # 1x219
            predict_result.append(predict_vector)
        ensemble_predict = [sum(sub_list) / len(sub_list) for sub_list in zip(*predict_result)] # cpu numpy list
        ensemble_predict = torch.tensor([item.cpu().detach().numpy() for item in ensemble_predict]).cuda() # to gpu tensor
        predicts = torch.argmax(ensemble_predict, dim=1)
        class_predict = predicts.item()
        result.append([image_name, class_predict])

    write_csv(data=result, csv_path=out_dir, save_name='submission')
    print(f'finish predicting! csv result is saved in {out_dir}')
