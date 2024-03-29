# [AICUP 2022] Competition-2022-Pytorch-Classification  
## TEAM_482: [Chi-Mao Fan](https://github.com/FanChiMao), [Yu-Chen Su](https://github.com/Modovado), [Wei-Hsiang Liao](https://github.com/zxc741852741)  

- [**尋找花中君子－蘭花種類辨識及分類**](https://tbrain.trendmicro.com.tw/Competitions/Details/20)  


<a href="https://tbrain.trendmicro.com.tw/Competitions/Details/20"><img src="https://i.imgur.com/Ubhj0LR.png" title="source: imgur.com" /></a>  

[![report](https://img.shields.io/badge/Supplementary-Report-yellow)](https://drive.google.com/drive/folders/1NzX75sgm8Z4br_NVP4SDZ80CjpjfUb_f?usp=sharing) [![download](https://img.shields.io/github/downloads/FanChiMao/Competition-2022-Pytorch-Orchid_Classification/total)](https://github.com/FanChiMao/Competition-2022-Pytorch-Orchid_Classification/releases/tag/v0.0) ![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2022-Pytorch-Orchid_Classification&label=visitors&countColor=%232ccce4&style=plastic) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TAlJB2QhbgE6fW-a3qph8bJr0blbQvzN?usp=sharing) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/52Hz/Orchid_classification_AICUP)  

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
We support the code of three different ensemble methods as following: 
<details>  
<summary><strong>Ensemble figures...</strong></summary>   
  
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
</details>    

However, due to the time limitation, we only use **Average ensemble** method to improve our performance. **Traditional ensemble** and **Resnet ensemble** cost about <u>8 hours</u> to test on 81710 images by our GTX GPU 1080Ti.😱  

- Average ensemble  
  Before predicting the results via **average ensemble**, please first check the parameters in [predict_ensemble.yaml](https://github.com/FanChiMao/Competition-2022-Pytorch-Orchid_Classification/blob/main/predict_ensemble.yaml) are all correctly set. And directly run:  
  ```
  python predict_ensemble.py
  ```

- Traditional ensemble  
  Train the ensemble mlp via running `train_ensemble_mlp.py`.

- Resnet ensemble  
  Train the res-ensemble net via running `faster_res_ensemble_train.py` and predict the output label using `faster_res_ensemble_test.py`.

## Final result  
  
- Score (accuracy)  
    - Public dataset: 90.00%  
    - Private dataset: 78.03%  
    - General final score[^1]: 81.63%  
    - Specific orchids: 96.15%  
  
        |                     |  Public set  |  Private set |  General final score|  
        | ------------------- | :----------: | :----------: | :-----------------: |  
        | Best accuracy       |      0.900077|      0.780395|          0.816300277|  
  

- Official final leaderboard  
    - Leaderboard: [leaderboard pdf](https://drive.google.com/file/d/1gx3oWq4HYiNtvudT_r7Gu3XkcVNJQmB6/view?usp=sharing)  
    - Registration teams: 743  
    - Participating teams: 275  
    - Our (TEAM_482) final rank: 18-th[^2]  
  

## Reference  
- https://github.com/rwightman/pytorch-image-models.git


## Contact us  
- Chi-Mao Fan (leader): qaz5517359@gmail.com  
- Yu-Chen Su:  qqsunmoonstar@gmail.com  
- Wei-Hsiang Liao: zxc741852741@gmail.com  

[^1]: General final score = 0.3xPublic + 0.7xPrivate  
[^2]: Winners depend on the baseline score where general final score > 0.79  


