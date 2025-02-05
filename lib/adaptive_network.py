import numpy as np
from torch import nn
import torch
from lib.network import Sampling_subnet, sample_allocation, PiABM_Net
import copy

class AdaCS(nn.Module):
    def __init__(self, stage, blocksize, subrates=[0.05, 0.05, 0.05, 0.05],
                 cr_ratios=[0.7, 0.7**2, 0.7**3]):
        super().__init__()
        self.stage = stage
        self.blocksize = blocksize
        self.cs_points_stage0 = int(np.round(blocksize * blocksize * subrates[0]))
        self.cs0 = Sampling_subnet(blocksize=blocksize, cs_points=self.cs_points_stage0)
        self.recon0 = PiABM_Net(blocksize=blocksize, cs_points=self.cs_points_stage0)

        cur_cs_points = self.cs_points_stage0
        self.cs_points_stage1 = int(np.round((blocksize * blocksize * np.sum(subrates[:2]) - cur_cs_points) / cr_ratios[0]))
        cur_cs_points += self.cs_points_stage1
        self.attention0 = sample_allocation(out_channels=self.cs_points_stage1, cr_ratio=cr_ratios[0])
        self.cs1 = Sampling_subnet(blocksize=blocksize, cs_points=self.cs_points_stage1)
        self.recon1 = PiABM_Net(blocksize=blocksize, cs_points=cur_cs_points)

        self.cs_points_stage2 = int(np.round((blocksize * blocksize * np.sum(subrates[:3]) - self.cs_points_stage0 -
                                             self.cs_points_stage1 * cr_ratios[0]) / cr_ratios[1]))
        cur_cs_points += self.cs_points_stage2
        self.attention1 = sample_allocation(out_channels=self.cs_points_stage2, cr_ratio=cr_ratios[1])
        self.cs2 = Sampling_subnet(blocksize=blocksize, cs_points=self.cs_points_stage2)
        self.recon2 = PiABM_Net(blocksize=blocksize, cs_points=cur_cs_points)

        self.cs_points_stage3 = int(np.round((blocksize * blocksize * np.sum(subrates[:4]) - self.cs_points_stage0 -
                                             self.cs_points_stage1 * cr_ratios[0] - self.cs_points_stage2 * cr_ratios[1]) / cr_ratios[2]))
        cur_cs_points += self.cs_points_stage3
        self.attention2 = sample_allocation(out_channels=self.cs_points_stage3, cr_ratio=cr_ratios[2])
        self.cs3 = Sampling_subnet(blocksize=blocksize, cs_points=self.cs_points_stage3)
        self.recon3 = PiABM_Net(blocksize=blocksize, cs_points=cur_cs_points)

    def forward(self, img):
        y, phiT_phi = self.cs0(img)
        y0 = copy.deepcopy(y.detach())
        x = self.recon0(y)
        delta_y_calA = y0 - self.cs0(x)[0]
        if self.stage >= 1:
            delta_y_calA = torch.linalg.norm(delta_y_calA, dim=1, ord=2) ** 2
            Amap1 = self.attention0(delta_y_calA)
            Mod_image, phiT_phi = self.cs1(img)
            y = torch.mul(Amap1, Mod_image)
            y1_m = copy.deepcopy(y.detach())
            y = torch.cat([y0, y], dim=1)
            y1 = copy.deepcopy(y.detach())
            x = self.recon1(y)
            # delta_y_0 = y0 - self.cs0(x)[0]
        if self.stage >= 2:
            delta_y_x_01 = torch.cat([(y0 - self.cs0(x)[0]),
                                          (y1_m - torch.mul(self.cs1(x)[0], Amap1))], dim=1)
            # vals_y_0 = torch.linalg.norm(delta_y_0, dim=1, ord=2) ** 2
            vals_y_x_01 = torch.linalg.norm(delta_y_x_01, dim=1, ord=2) ** 2
            scaled_vals_y_x_01 = vals_y_x_01 / (Amap1[:, 0, :, :]+1)
            Amap2 = self.attention1(scaled_vals_y_x_01)
            Mod_image, phiT_phi = self.cs2(img)
            y = torch.mul(Amap2, Mod_image)
            y2_m = copy.deepcopy(y.detach())
            y = torch.cat([y1, y], dim=1)
            y2 = copy.deepcopy(y.detach())
            x = self.recon2(y)
            # delta_y_0 = y0 - self.cs0(x)[0]
        if self.stage >= 3:
            # vals_y_0 = torch.linalg.norm(delta_y_0, dim=1, ord=2) ** 2
            delta_y_x_012 = torch.cat([(y0 - self.cs0(x)[0]),
                                         (y1_m - torch.mul(self.cs1(x)[0], Amap1)),
                                           (y2_m - torch.mul(self.cs2(x)[0], Amap2))], dim=1)
            vals_y_x_012 = torch.linalg.norm(delta_y_x_012, dim=1, ord=2) ** 2
            scaled_vals_y_x_012 = vals_y_x_012 / (1+Amap1[:, 0, :, :]+Amap2[:, 0, :, :])
            Amap3 = self.attention2(scaled_vals_y_x_012)
            Mod_image, phiT_phi = self.cs3(img)
            y = torch.mul(Amap3, Mod_image)
            y = torch.cat([y2, y], dim=1)
            x= self.recon3(y)
        return x, phiT_phi