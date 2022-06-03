import csv
import os
import pandas as pd
import statistics
from tqdm import tqdm


def write_csv(data=None, csv_path=None, save_name='submission'):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    fieldnames = ['filename', 'category']
    with open(csv_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for i in range(len(data)):
            name = data[i][0]
            predict = data[i][1]
            w.writerow([name, predict])
    f.close()