# Competition-2022-Pytorch-Orchid_Classification  
## [Chi-Mao Fan](https://github.com/FanChiMao), [Yu-Chen Su](https://github.com/Modovado), [Wei-siang Liao](https://github.com/zxc741852741)  
<a href="[https://imgur.com/mc4Di1O](https://tbrain.trendmicro.com.tw/Competitions/Details/20)"><img src="https://i.imgur.com/mc4Di1O.png" title="source: imgur.com" /></a>  

- [尋找花中君子－蘭花種類辨識及分類](https://tbrain.trendmicro.com.tw/Competitions/Details/20)  

```
├── README.md    

training
├── train.yaml
├── train.py
├── utils
├── dataset

testing
├── predict.py
├── result

```

## Installation
The model is built in PyTorch 1.8.0 and tested on Windows10 environment  
(Python: 3.8, CUDA: 10.2, cudnn: 7.6).  

For installing, follow these intructions
```
conda create -n pytorch python=3.8  
conda activate pytorch  
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch  
conda install -c conda-forge tensorboardx
pip install git+https://github.com/rwightman/pytorch-image-models.git
pip install -r requirements.txt
```

## Dataset  
You can refer the [**README.md**](dataset/README.md) to prepare the dataset.  

## Train each classifier  
Set hyperparameters and revelent training path in [**train.yaml**](train.yaml) and simply run [**train.py**](train.py).  

## Predict via each trained classifier  
You can download our pretrained model from [**pretrained**](./pretrained).  
To predict the orchid images by single classifier, see [**predict.py**](predict.py) and run:  
```
python predict.py --model model_name --input_dir images_folder_path --result_dir save_csv_here --weights path_to_models
```

## Ensemble strategy  
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/FPR0WoU.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/xMVL6N1.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/xMVL6N1.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Half Wavelet Attention Block (HWAB)</b></p></td>
    <td><p align="center"> <b>Resizing Block (Pixel Shuffle)</b></p></td>
    <td><p align="center"> <b>Resizing Block (Pixel Shuffle)</b></p></td>
  </tr>
</table>


