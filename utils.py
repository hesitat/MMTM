# -*- coding: utf-8 -*-


from lifelines.utils import concordance_index as ci
import torch

from lifelines.utils import concordance_index

import numpy as np
# %matplotlib inline

from torch import nn







class CoxRegression(nn.Module):
    def __init__(self, input_size):
        super(CoxRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y1 = self.relu(self.linear(x))
        y2 = self.linear2(y1)
        y3 = self.relu(self.linear3(y2))
        y4 = self.linear4(y3)
        return y4


def concordance_index(y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, y_pred, e)
    return ci_value


class DeepCox_Loss(nn.Module):
    def __init__(self):
        super(DeepCox_Loss, self).__init__()

    def forward(self, y_predict, t):
        y_pred_list = y_predict.view(-1)
        y_pred_exp = torch.exp(y_pred_list)
        t_list = t.view(-1)
        t_E = torch.gt(t_list, 0)

        y_pred_cumsum = torch.cumsum(y_pred_exp, dim=0)
        y_pred_cumsum_log = torch.log(y_pred_cumsum)
        loss1 = -torch.sum(y_pred_list.mul(t_E))

        loss2 = torch.sum(y_pred_cumsum_log.mul(t_E))

        loss = (loss1 + loss2) / torch.sum(t_E)
        return loss