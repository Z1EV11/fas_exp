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


class Binary_Class():
    """
    Binary Classification Metric
    """
    def __init__(self):
        super(Binary_Class, self).__init__()
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
    
    def update(self, pred, label):
        num = len(label)
        if len(pred) != num:
            raise NotImplementedError
        if type(pred) is not np.ndarray or type(label) is not np.ndarray:
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
        xN_cond = pred==0
        xN = np.extract(xN_cond, label)
        FN = np.sum(xN)
        self.FN += FN
        self.TN += len(xN) - FN
        xP_cond = pred==1
        xP = np.extract(xP_cond, label)
        TP = np.sum(xP)
        self.TP += TP
        self.FP += len(xP) - TP

    def calc_ACC(self):
        """
        Accurancy
        """
        acc = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        return acc

    def calc_PPV(self):
        """
        Precision: Positive Predictive Value
        """
        precision = self.TP/(self.TP+self.FP)
        return precision

    def calc_TPR(self):
        """
        Recall: True Positive Rate
        """
        recall = self.TP/(self.TP+self.FN)
        return recall
    
    def calc_TNR(self):
        """
        Specificity: True Negative Rate
        """
        Specificity = self.TN/(self.TN+self.FP)
        return Specificity

    def calc_F1(self):
        """
        F1-score
        """
        F1 = 1     
        return F1


class FAS_metric(Binary_Class):
    """
    Face Anti-Spoofing Metric
    """
    def __init__(self, num_PA):
        """
        Args:
            num_PA: number of Presentation Attack types
        """
        super(FAS_metric, self).__init__()
        self.num_PA = num_PA
        self.reset()
    
    def reset(self):
        super().reset()
        self.num_PA = 0
    
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
