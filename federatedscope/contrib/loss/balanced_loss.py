from federatedscope.register import register_criterion
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class BalancedLoss(torch.nn.modules.loss._Loss):
    """
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    reference: https://github1s.com/fcakyon/balanced-loss/blob/HEAD/balanced_loss/losses.py#L82-L92
    reference: https://github1s.com/vandit15/Class-balanced-loss-pytorch/blob/HEAD/class_balanced_loss.py

    Args
      # samples_per_cls: A python list of size [num_classes].
      loss_type: string. One of "focal_loss", "cross_entropy", "binary_cross_entropy", "softmax_binary_cross_entropy".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    def __init__(self, loss_type, beta=0, fl_gamma=2, samples_per_class=None):
        super(BalancedLoss, self).__init__()

        self.loss_type = loss_type
        self.beta = beta
        # beta -> 0 corresponds to no-reweighting, beta -> 1 corresponds to re-weighting by inverse class frequency
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class

    def forward(self, logits, labels, beta=None, fl_gamma=None, samples_per_class=None):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if samples_per_class is None:  # else, use samples_per_class
            if self.samples_per_class is None:
                # Warning("Using class-balanced loss, but the samples_per_class is not provided, "
                #         "then the class statistics within one batch will be used.")  # verbose print
                samples_per_class = np.bincount(labels.cpu(), minlength=num_classes)
                samples_per_class = samples_per_class.clip(1, np.Inf)  # to avoid the number of objs corresponding to some class is zero.
            else:
                samples_per_class = self.samples_per_class

        beta = beta if beta is not None else self.beta

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        weights = torch.tensor(weights, device=logits.device).float()

        if self.loss_type != "cross_entropy":
            weights = weights.unsqueeze(0)
            weights = weights.repeat(batch_size, 1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, num_classes)

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=fl_gamma if fl_gamma is not None else self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)  # sigmoid
        elif self.loss_type == "softmax_binary_cross_entropy":
            cb_loss = F.binary_cross_entropy(input=logits.softmax(dim=-1), target=labels_one_hot, weight=weights)  # softmax

        return cb_loss


def call_bl_criterion(type, device, **kwargs):
    if type == 'balanced_fl':
        return BalancedLoss("focal_loss", **kwargs).to(device)
    if type == 'balanced_ce':
        return BalancedLoss("cross_entropy", **kwargs).to(device)
    if type == 'balanced_bce':
        return BalancedLoss("binary_cross_entropy", **kwargs).to(device)
    if type == 'balanced_softmax_bce':
        return BalancedLoss("softmax_binary_cross_entropy", **kwargs).to(device)


register_criterion('balanced_loss', call_bl_criterion)
