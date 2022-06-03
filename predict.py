from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import argparse
from utils.model_utils import *
import os
from natsort import natsorted
from glob import glob
from utils.csv_utils import write_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict orchid class')
    parser.add_argument('--model', default='vit', type=str, help='model name [vit, beit, convnext, swin]')
    parser.add_argument('--input_dir', default='dataset/dataset_category_split/test/0', type=str, help='Input images')
    parser.add_argument('--result_dir', default='./results', type=str, help='Directory for results')
    parser.add_argument('--weights', default='pretrained/convnext_fold1_best_acc.pth', type=str, help='Path to weights')

    args = parser.parse_args()

    inp_dir = args.input_dir
    out_dir = args.result_dir
    os.makedirs(out_dir, exist_ok=True)

    files = natsorted(glob(os.path.join(inp_dir, '*.JPG')) + glob(os.path.join(inp_dir, '*.PNG')))
    model = build_model(args.model)
    model.cuda()

    load_checkpoint(model, args.weights)
    model.eval()

    print('Start predicting......')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=480, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        # transforms.RandomCrop(384)
    ])
    result = []
    for i, file_ in enumerate(tqdm(files)):
        image_name = os.path.split(file_)[-1]
        img = Image.open(file_).convert('RGB')
        input_ = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            predict_result = model(input_)
        predicts = torch.argmax(predict_result, dim=1)
        # predicts = torch.topk(predict_result, 10).indices
        class_predict = predicts.item()
        result.append([image_name, class_predict])

    write_csv(data=result, csv_path=args.result_dir)
    print('finish !')