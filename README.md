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
pip install -r requirements.txt
```
