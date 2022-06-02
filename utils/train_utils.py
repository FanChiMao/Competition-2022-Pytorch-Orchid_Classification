import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import os

# loss thing
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# metric thing
class MetricMeter:  # for metric
    def reset(self):
        list_size = self.label
        matrix_size = (self.label, self.label)

        # mean value
        self.o_acc = 0
        self.prec_mean = 0
        self.rec_mean = 0
        self.f1_mean = 0
        self.iou_mean = 0

        # list
        self.o_acc_list = np.zeros(list_size)
        self.prec_list = np.zeros(list_size)
        self.rec_list = np.zeros(list_size)
        self.f1_list = np.zeros(list_size)
        self.iou_list = np.zeros(list_size)

        # confusion matrix
        self.cm = np.zeros(matrix_size)

    def __init__(self, label=1):
        self.label = label
        self.reset()

    def update(self, value: np.array, n=1):
        self.cm += value

        # tp, tn, fp, fn
        tp = np.diag(self.cm)
        fp = np.sum(self.cm, axis=0) - np.diag(self.cm)
        fn = np.sum(self.cm, axis=1) - np.diag(self.cm)
        tn = np.sum(self.cm) - np.sum(self.cm, axis=0) - np.sum(self.cm, axis=1) + np.diag(self.cm)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.o_acc = (tp / (tp + fp + tn + fn)).sum()
            self.prec_mean = (tp / (tp + fp)).mean()
            self.rec_mean = (tp / (tp + fn)).mean()
            self.f1_mean = (tp / (tp + 0.5 * (fp + fn))).mean()
            self.iou_mean = (tp / (tp + fp + fn)).mean()

            self.o_acc_list = tp / (tp + fp + tn + fn)
            self.prec_list = tp / (tp + fp)
            self.rec_list = tp / (tp + fn)
            self.f1_list = tp / (tp + 0.5 * (fp + fn))
            self.iou_list = tp / (tp + fp + fn)


def metric_(predict: torch.Tensor, target: torch.Tensor, num_classes=5):
    matrix_size = (num_classes, num_classes)

    # confusion index
    cm_index = np.arange(num_classes)

    # tensor to numpy & flatten
    predict = predict.detach().cpu().numpy().astype('int').ravel()
    target = target.detach().cpu().numpy().astype('int').ravel()

    # unique
    target_unique = np.unique(target)

    # check target `any` index is in cm_index
    # if not np.isin(target_unique, cm_index).any():
    #
    #     # prevent `ValueError: At least one label specified must be in y_true`
    #     cm = np.zeros(matrix_size)
    #
    # else:
    cm = confusion_matrix(target, predict, labels=cm_index)

    return cm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)