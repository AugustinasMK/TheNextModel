import torch
import torch.nn as nn


# Since we are using the ViT-Large model, we will use 16 groups
# p_params: pooling parameters, the number of pooling parameters is the same with groups
class GGeM(nn.Module):
    def __init__(self, groups=16, eps=1e-6):
        super().__init__()
        self.groups = groups
        self.p_params = nn.Parameter(torch.ones(groups) * 3)
        self.eps = eps

    def forward(self, x):
        x = x[:, 1:, :]  # Remove class token

        batch_size, tokens, dimensions = x.shape
        e = dimensions // self.groups
        x = x.reshape((batch_size, tokens, e, self.groups))

        x = x.clamp(min=self.eps).pow(self.p_params)
        x = x.mean(dim=1)
        x = x.pow(1. / self.p_params)

        x = x.reshape((batch_size, dimensions))
        return x


class ClassToken(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 0]
