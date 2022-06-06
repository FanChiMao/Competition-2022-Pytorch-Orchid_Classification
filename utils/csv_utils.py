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
    import csv
import os
import pandas as pd
from statistics import mode, StatisticsError
import numpy


def write_csv(data=None, csv_path=None, save_name='submission'):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    fieldnames = ['filename', 'category']
    with open(csv_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for i in range(len(data)):
            name = data[i][0]
            predict = int(data[i][1])
            w.writerow([name, predict])
    f.close()

def csv_vote(csv_0, csv_1, csv_2, main: int=0):

    filename = pd.read_csv(csv_0)['filename']
    label_1 = pd.read_csv(csv_0)['category']
    label_2 = pd.read_csv(csv_1)['category']
    label_3 = pd.read_csv(csv_2)['category']
    
    len_pd = len(label_1)
    y = numpy.vstack((label_1, label_2, label_3)).T
    output = []

    for i, x in enumerate(y):
        try:
            most_common = mode(x)
        except StatisticsError:
            most_common = x[main]

        output.append(most_common)

    result = [[x, y] for x, y in zip(filename, output)]
    write_csv(data=result, csv_path='../results', save_name='submission')



    

