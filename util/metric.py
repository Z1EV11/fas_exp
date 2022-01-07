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


class BinaryClassMetric():
    """
    Binary Classification Metric
        Positive: 1
        Negative: 0
    """
    def __init__(self):
        super(BinaryClassMetric, self).__init__()
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
        if len(pred) != len(label):
            raise NotImplementedError
        if type(pred) is not np.ndarray or type(label) is not np.ndarray:
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
        xN_cond = pred==0
        xN = np.extract(xN_cond, label)
        fN = np.sum(xN)
        self.FN += fN
        self.TN += len(xN) - fN
        xP_cond = pred==1
        xP = np.extract(xP_cond, label)
        tP = np.sum(xP)
        self.TP += tP
        self.FP += len(xP) - tP

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


class FASMetric(BinaryClassMetric):
    """
    Face Anti-Spoofing Metric
    """
    def __init__(self, num_PA=1):
        """
        Args:
            num_PA: number of Presentation Attack types
        """
        super(FASMetric, self).__init__()
        self.num_PA = num_PA
        self.reset()
    
    def reset(self):
        super().reset()
        self.num_PA = 1

    # def update(self, pred, label):
    #     if len(pred) != len(label):
    #         raise NotImplementedError
    #     if type(pred) is not np.ndarray or type(label) is not np.ndarray:
    #         pred = pred.cpu().numpy()
    #         label = label.cpu().numpy()
    #     super().update(pred, label)
    #     num_live = np.sum(label)
    #     num_fake = len(label) - num_live
    #     self.num_live += num_live
    #     self.num_fake += num_fake
    
    def calc_ACER(self):
        """
        Average Classification Error Rate
            Live: 0
            Fake: 1
        Returns:
            ACER: (max{APCER}+BPCER)/2
            APCER: Attack Presentation Classification Error Rate == max{FAR}
            BPCER: Bona fide Presentation Classification Error Rate == FRR
        """
        apcer = self.FN/self.num_fake
        bpcer = self.FP/self.num_live
        acer = (apcer+bpcer)/2.0
        return acer, apcer, bpcer

    def calc_HTER(self):
        """
        Half Total Error Rate(HTER)
        FAR(False Acceptation Rate): Fake is judged as Live
        FRR(False Rejection Rate):  Live is judged as Fake
        """
        far = self.FP / (self.FP + self.TN)
        frr = self.FN / (self.TP + self.FN)
        hter = (far + frr) / 2.0
        return hter, far, frr
