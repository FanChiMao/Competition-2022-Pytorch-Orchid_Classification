# Competition-2022-Pytorch-Orchid_Classification  
## TEAM_482: [Chi-Mao Fan](https://github.com/FanChiMao), [Yu-Chen Su](https://github.com/Modovado), [Wei-Hsiang Liao](https://github.com/zxc741852741)  

- [**尋找花中君子－蘭花種類辨識及分類**](https://tbrain.trendmicro.com.tw/Competitions/Details/20)  
<a href="[https://imgur.com/mc4Di1O](https://imgur.com/Ubhj0LR)(https://tbrain.trendmicro.com.tw/Competitions/Details/20)"><img src="https://i.imgur.com/Ubhj0LR.png" title="source: imgur.com" /></a>  

[![download](https://img.shields.io/github/downloads/FanChiMao/Competition-2022-Pytorch-Orchid_Classification/total)](https://github.com/FanChiMao/Competition-2022-Pytorch-Orchid_Classification/releases/tag/v0.0) ![visitors](https://visitor-badge.glitch.me/badge?page_id=FanChiMao/Orchid_AICUP) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TAlJB2QhbgE6fW-a3qph8bJr0blbQvzN?usp=sharing) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/52Hz/Orchid_classification_AICUP)  

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
    <td> <img src = "https://i.imgur.com/g4GREcK.jpg" width="400"> </td>
    <td> <img src = "https://i.imgur.com/WA4jq5G.jpg" width="400"> </td>
    <td> <img src = "https://i.imgur.com/wlnXdpx.jpg" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Average ensemble</b></p></td>
    <td><p align="center"><b>Traditional ensemble</b></p></td>
    <td><p align="center"><b>Resnet ensemble</b></p></td>
  </tr>
</table>


## Predict result  
<details>  
<summary><strong>More details...</strong></summary>   
  
- Score:  

    - Best score  
  
        |                     |  Public set  |  Private set |  General final score|  
        | ------------------- | :----------: | :----------: | :-----------------: |  
        | Best accuracy       |      0.900077|      0.780395|          0.816300277|  
  
        Note: General final score = 0.3xPublic + 0.7xPrivate  
          

- Official final leaderboard  
    - PDF leaderboard: https://drive.google.com/file/d/1XEMtN1nZEf0kqwAbFIGAT0S_sKUQss-K/view?usp=sharing  
  
    - 
  
</details>    
  

## Reference  
- https://github.com/rwightman/pytorch-image-models.git


## Contact us  
- Chi-Mao Fan: qaz5517359@gmail.com  
- Yu-Chen Su:  qqsunmoonstar@gmail.com
- Wei-Hsiang Liao: zxc741852741@gmail.com
