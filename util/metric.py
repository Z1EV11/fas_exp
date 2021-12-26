import numpy as np
import torch
from torch.nn.functional import leaky_relu


class AvgMeter(object):
    """
        Computes and stores the average and current value
    """
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


class Accuracy(AvgMeter):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.reset()

    def calc_acc(self, pred, label):
        """
        Accuracy
        Args:
            perd: tensor.
            label: tensor.
        """
        err = torch.mean(torch.sum(torch.abs(label-pred)) / len(label))
        acc = 1 - err
        return acc
    
    def get_avg_acc(self):
        return self.avg


class Classification(AvgMeter):
    def __init__(self):
        super(Classification, self).__init__()
        self.acc = Accuracy()
        self.reset()
    
    def update(self, metric, val, n=1):
        if metric == 'accurancy':
            self.acc.update(val, n)

    def calc_precision(self):
        """
        Precision
        """
        precision = 1
        return precision

    def calc_recall(self):
        """
        Recall
        """
        recall = 1
        return recall

    def clac_F1_socre(self):
        """
        F1-score
        """
        F1 = 1     
        return F1


class FAS_metric(AvgMeter):
    def __init__(self, num_PA):
        """
        Args:
            num_PA: number of Presentation Attack types
        """
        super(FAS_metric, self).__init__()
        self.num_PA = num_PA
        self.reset()
    
    def calc_acer(self, pred, label):
        """
        Average Classification Error Rate
        Returns:
            ACER: (APCER+BPCER)/2
            APCER: Attack Presentation Classification Error Rate
            BPCER: Bonafide Presentation Classification Error Rate
        """
        apcer = 1
        bpcer = 1
        acer = (apcer+bpcer)/2
        return acer, apcer, bpcer
    
    def calc_apcer(self, pred, label):
        apcer = 1
        return apcer

    def calc_bpcer(self, pred, label):
        bpcer = 1
        return bpcer
