# code from https://github.com/ts-kim/RevIN, with minor modifications

'''
Purpose of RevIN
Handling Non-Stationarity:
- Traditional normalization methods (e.g., z-score normalization) are applied globally, using the mean and standard deviation of the entire dataset. 
 This approach assumes that the statistical properties of the data are constant over time.
- Time series data, especially real-world data, often exhibit non-stationary behavior where these statistical properties change over time. RevIN 
 normalizes each instance (e.g., each time series segment) separately, adapting to local changes in the data's distribution.
- RevIN normalizes each instance separately, making it more effective in handling these changes compared to BatchNorm, which normalizes across the batch.

Mitigating Distribution Shifts:
- In many practical scenarios, the training data and the test data can come from different distributions due to changes over time (distribution shift). 
 RevIN helps the model to be more robust to such shifts by normalizing the data instance-wise during both training and inference.

Reversibility:
- After making predictions, the data needs to be transformed back to its original scale for interpretation and use. RevIN ensures that this process is 
 reversible, allowing the model to operate on normalized data while still being able to produce predictions in the original scale.
'''

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
