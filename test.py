import argparse
import torch
from torch.autograd import Variable
import numpy as np
from lib.adaptive_network import AdaCS
import os
import cv2
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder, PSNR
import copy

parser = argparse.ArgumentParser(description="PyTorch code of AdaCS_TPAMI_2024 by Chenxi Qiu")
parser.add_argument("--model", default="weights/AdaCS/stage3/net_epoch_best.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./testsets", type=str, help="dataset path")
parser.add_argument("--test_name", default="Set11", type=str, help="dataset name")
parser.add_argument('--block_size', default=8, type=int, help='CS block size')
parser.add_argument('--test_stage', default=3, type=int, help='test stage, 0 for rate 5%, 1 for rate 10%, ...')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_list

model = AdaCS(stage=opt.test_stage, blocksize=opt.block_size)
if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

test_set = TestDatasetFromFolder(os.path.join(opt.dataset, opt.test_name), blocksize=opt.block_size)
test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

psnr_list, ssim_list = [], []
with torch.no_grad():
    for im_path in test_set.image_filenames:
        data = cv2.imread(im_path, 0)
        data = torch.from_numpy(data)
        [row, col] = data.shape
        data = data.type(torch.FloatTensor) / 255
        gt = copy.deepcopy(data)
        row_pad = opt.block_size - np.mod(row, opt.block_size)
        col_pad = opt.block_size - np.mod(col, opt.block_size)
        data = torch.cat([data, torch.zeros([row, col_pad])], dim=1) if col_pad < opt.block_size else data
        if col_pad < opt.block_size:
            data = torch.cat([data, torch.zeros([row_pad, col + col_pad])], dim=0)
        else:
            data = torch.cat([data, torch.zeros([row_pad, col])], dim=0) if row_pad < opt.block_size else data
        data = torch.unsqueeze(data, 0)
        data = torch.unsqueeze(data, 0)

        im_input = Variable(data)
        model = model.cuda()
        im_input = im_input.cuda()
        res, _ = model(im_input)
        res = res.cpu()
        res = res.data[0].numpy().astype(np.float32)
        res = res * 255.
        res[res < 0] = 0
        res[res > 255.] = 255.
        res = res[0, :, :]
        psnr_predicted = PSNR(gt.numpy() * 255., res[:row, :col],shave_border=0)
        ssim_predicted = SSIM(gt.numpy() * 255., res[:row, :col], data_range=255)
        psnr_list.append(psnr_predicted)
        ssim_list.append(ssim_predicted)

print("Test Set= %s, PSNR = %.4e, SSIM = %.4e"% (opt.test_name, np.mean(psnr_list), np.mean(ssim_list)))
