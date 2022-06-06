# Dataset  

## Download the training images  
Download [official training images](https://tbrain.trendmicro.com.tw/Competitions/Details/20) or download from our [google cloud](https://drive.google.com/drive/folders/1j2DS-WhUs0ezzHVzNOEgZ4h9_1oPPqZh?usp=sharing) (same as official images). Place these images to folder `dataset_original`. And place `label.csv` in this root.  

## Data augmentation  
Run the data augmentation code ([data_augmentation.py](../utils/data_augmentation.py))  

## Tree  
Your directory structure should look like this  

<details>  
<summary>Training tree...</summary>   
  
  ```
  dataset                   # datasets root
    ├── dataset_original    # original training images  
    |    ├── zthjatja.jpg             
    |    ├── srtjrary.jpg   
    |    ├── ...
    |    └── qaertaeg.jpg
    |
    ├── new_dataset         # data augmentation and preprocess images
    |    ├── 1.jpg             
    |    ├── 2.jpg   
    |    ├── ...
    |    ├── 17520.jpg
    |    └── label.csv      # auto generated after running utils/data_augmentation.py
    |
    └── label.csv           # original label csv
  
                   
  ```  
