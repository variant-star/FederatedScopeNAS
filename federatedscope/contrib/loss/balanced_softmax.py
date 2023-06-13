from federatedscope.register import register_criterion
import torch
import torch.nn.functional as F
import numpy as np


class BalancedSoftmax(torch.nn.modules.loss._Loss):
    """
    Balanced Softmax Loss
    References: https://github1s.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/HEAD/loss/BalancedSoftmaxLoss.py
    """
    def __init__(self, samples_per_class=None):
        super(BalancedSoftmax, self).__init__()
        self.samples_per_class = samples_per_class

    def forward(self, logits, labels, samples_per_class=None):
        """
        Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          sample_per_class: A int tensor of size [no of classes].
          reduction: string. One of "none", "mean", "sum"
        Returns:
          loss: A float tensor. Balanced Softmax Loss.
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        if samples_per_class is None:  # else, use samples_per_class
            if self.samples_per_class is None:
                # Warning("Using class-balanced loss, but the samples_per_class is not provided, "
                #         "then the class statistics within one batch will be used.")  # verbose print
                samples_per_class = np.bincount(labels.cpu(), minlength=num_classes)
                samples_per_class = samples_per_class.clip(1, np.Inf)  # to avoid the number of objs corresponding to some class is zero.
            else:
                samples_per_class = self.samples_per_class

        spc = torch.tensor(samples_per_class, device=logits.device).float()
        spc = spc.unsqueeze(0).expand(batch_size, -1)
        logits = logits + spc.log()

        return F.cross_entropy(input=logits, target=labels)


def call_my_criterion(type, device):
    if type == 'balanced_softmax':
        return BalancedSoftmax().to(device)


register_criterion('balanced_softmax', call_my_criterion)
