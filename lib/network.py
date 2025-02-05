from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from lib.basic_blocks import Block, UpSamplingBlock, DownSamplingBlock


class sample_allocation(nn.Module):
    def __init__(self, out_channels, cr_ratio):
        super().__init__()
        self.cr_ratio = cr_ratio
        self.out_channels = out_channels
    def forward(self, vals):
        B, H, W = vals.shape
        temp = torch.reshape(vals, [B, H*W])
        num_save_points = int(round(H * W *  self.cr_ratio))
        [divide_point, _] = torch.kthvalue(temp, H * W- num_save_points, dim=1)
        d1 = torch.unsqueeze(torch.unsqueeze(divide_point, -1), -1)
        vals = (vals - d1)
        vals = vals / torch.max(torch.abs(vals) * 2)
        vals = torch.ceil(vals)
        vals = torch.unsqueeze(vals, dim=1)
        out = repeat(vals, 'b c h w -> b (c repeat) h w', repeat=self.out_channels)
        return out


class Sampling_subnet(nn.Module):
    def __init__(self, blocksize, cs_points):
        super(Sampling_subnet, self).__init__()
        self.blocksize = blocksize
        self.cs_points = cs_points
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(cs_points,
                                                                 blocksize * blocksize)))
    def forward(self, img):
        Phi_ = torch.nn.functional.normalize(self.Phi, p=2, dim=0)
        phiT_phi = torch.mm(torch.transpose(Phi_, 0, 1), Phi_)
        PhiWeight = Phi_.contiguous().view(self.cs_points, 1, self.blocksize, self.blocksize)
        y = F.conv2d(img, PhiWeight, padding=0, stride=self.blocksize, bias=None)
        return y, phiT_phi

class PiABM_Net(nn.Module):
    def __init__(self, blocksize, cs_points, num_features=[96, 64, 32], scales=[4, 2, 1], dd=5):
        super(PiABM_Net, self).__init__()
        self.blocksize = blocksize
        self.cs_points = cs_points
        self.scales = scales
        self.convnext = Block(dim=cs_points)
        for r, n_ch in enumerate(num_features):
            for c in range(dd - 1):
                setattr(self, f'convnext_{r}_{c}', Block(dim=n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(num_features[:-1], num_features[1:])):
            for c in range(0, int(dd), 2):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(num_features[1:], num_features[:-1])):
            for c in range(1, int(dd), 2):
                setattr(self, f'down_{r+1}_{c}', DownSamplingBlock(in_ch, out_ch))

        self.upsampling = nn.Conv2d(cs_points,
                                    (blocksize // scales[0]) * (blocksize // scales[0]) * num_features[0],
                                    kernel_size=1, stride=1,padding=0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features[0], num_features[0], kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.finetune = Block(dim=num_features[-1])
        self.conv_out = nn.Conv2d(num_features[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.convnext(x)
        x = self.upsampling(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)',
                      p1=self.blocksize // self.scales[0], p2=self.blocksize // self.scales[0])

        x00 = self.conv1(x)
        x10 = self.up_0_0(x00)
        x20 = self.up_1_0(x10)

        x21 = self.convnext_2_0(x20)
        x11 = self.convnext_1_0(x10) + self.down_2_1(x21)
        x01 = self.convnext_0_0(x00) + self.down_1_1(x11)

        x02 = self.convnext_0_1(x01)
        x12 = self.convnext_1_1(x11) + self.up_0_2(x02)
        x22 = self.convnext_2_1(x21) + self.up_1_2(x12)

        x23 = self.convnext_2_2(x22)
        x13 =  self.convnext_1_2(x12) + self.down_2_3(x23)
        x03 = self.convnext_0_2(x02) + self.down_1_3(x13)

        x04 = self.convnext_0_3(x03)
        x14 = self.convnext_1_3(x13) + self.up_0_4(x04)
        x24 = self.convnext_2_3(x23) + self.up_1_4(x14)

        out = self.conv_out(self.finetune(x24))

        return out


