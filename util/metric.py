import numpy as np
import torch

class Metric():
    def __init__(self, acc=0.0, acer=0.0):
        """
        Args:
            acc: Accuracy
            acer: Average Classification Error Rate
        """
        self.acc = acc
        self.acer = acer

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

    def update(self, pred, label):
        acc = self.clac_acc(pred, label)
        self.acc = self.acc + torch.mean(acc)
        pass