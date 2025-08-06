# -*- coding: utf-8 -*-


import torch

# %matplotlib inline

from torch import nn

from utils import CoxRegression


class MMTM_mitigate(nn.Module):
    def __init__(self,
                 x0,
                 x1,
                 ratio,
                 device=torch.device("cpu"),
                 SEonly=False,
                 ):
        super(MMTM_mitigate, self).__init__()

        # ratio是压缩的大小 x0,x1是维度大小 20468和20156
        dim = x0 + x1
        dim_out = int(2 * dim / ratio)

        self.running_avg_weight_visual = torch.zeros(x0).to(device)
        self.running_avg_weight_skeleton = torch.zeros(x0).to(device)
        self.step = 0

        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, x0)
        self.fc_skeleton = nn.Linear(dim_out, x1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.cox = CoxRegression(dim)

    def forward(self,
                visual,
                skeleton,
                return_scale=False,
                return_squeezed_mps=False,
                turnoff_cross_modal_flow=False,
                average_squeezemaps=None,
                caring_modality=0,
                ):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))

        # 按第二个维度拼接
        squeeze = torch.cat(squeeze_array, 1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        self.running_avg_weight_visual = (vis_out.mean(0) + self.running_avg_weight_visual * self.step).detach() / (
                self.step + 1)
        self.running_avg_weight_skeleton = (vis_out.mean(0) + self.running_avg_weight_skeleton * self.step).detach() / (
                self.step + 1)

        self.step += 1

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        # 把输出拼接到一起
        x = torch.cat([visual * vis_out, skeleton * sk_out], 1)
        # 再经过一个线性层

        hazards = self.cox(x)

        return   visual * vis_out, skeleton * sk_out,hazards