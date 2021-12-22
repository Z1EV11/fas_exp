import numpy as np
import torch


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
        super(AvgMeter, self).__init__()
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

    def calc_precision(self):
        pass

    def calc_recall(self):
        pass

    def clac_F1_socre(self):
        pass


class FAS_metric(AvgMeter):
    def __init__(self, num_pa):
        """
        Args:
            num_pa: number of Presentation Attack types
        """
        super(AvgMeter, self).__init__()
        self.num_pa = num_pa
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
