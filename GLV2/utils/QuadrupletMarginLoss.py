import torch.nn as nn
from torch import Tensor
from torch.nn.functional import triplet_margin_loss


class QuadrupletMarginLoss(nn.Module):
    def __init__(self, margin: float = 1.0, alpha: float = 0.5, p: float = 2., eps: float = 1e-6, swap: bool = False,
                 reduction: str = 'mean'):
        super(QuadrupletMarginLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, semipositive: Tensor, negative: Tensor) -> Tensor:
        loss1 = triplet_margin_loss(anchor, positive, semipositive, margin=self.alpha * self.margin, p=self.p,
                                    eps=self.eps, swap=self.swap, reduction=self.reduction)
        loss2 = triplet_margin_loss(anchor, semipositive, negative, margin=(1.0 - self.alpha) * self.margin, p=self.p,
                                    eps=self.eps, swap=self.swap, reduction=self.reduction)
        return loss1 + loss2
