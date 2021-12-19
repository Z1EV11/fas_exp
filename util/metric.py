import numpy as np
import torch

class Metric():
    def __init__(self, acc=0.0, acer=0.0):
        """
        Args:
            avg_acc: Average Accuracy
            acer: Average Classification Error Rate
            sum: Error Sum
            count: Count of Batchs
        """
        self.count = 0
        self.acc_sum = 0
        self.avg_acc = 0
        self.acer = 0

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
        # acer = (apcer+bpcer)/2
        acer = self.sum/self.count
        return acer, apcer, bpcer

    def update(self, pred, label):
        acc = self.calc_acc(pred, label)
        self.acc_sum = self.acc_sum + acc
        self.count += 1

    def get_avg_acc(self):
        avg_acc = self.acc_sum/self.count
        return avg_acc

def calc_score(output, threshould=0.5):
    """
    Convert output estimation to spoof/live prediction
    """
    r = output[1].cpu()
    with torch.no_grad():
        output_binary_1 = r.data.numpy().flatten()
        score= np.mean(output_binary_1)
    return score