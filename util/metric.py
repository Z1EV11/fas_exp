import numpy as np
import torch

def get_acc(pred, label):
    """
    Accuracy
    Args:
        perd: tensor.
        label: tensor.
    """
    err_rate = torch.mean(torch.sum(torch.abs(label-pred)) / len(label))
    acc = 1-err_rate
    return acc

def get_precision():
    pass

def get_recall():
    pass

def get_F1_socre():
    pass

def get_apcer():
    """
    Attack Presentation Classification Error Rate
    """
    return 1

def get_bpcer():
    """
    Bonafide Presentation Classification Error Rate
    """
    return 1

def get_acer():
    """
    Average Classification Error Rate
    """
    apcer = get_apcer()
    bpcer = get_bpcer()
    acer = (apcer+bpcer)/2
    return acer

def update_metric():
    return 1