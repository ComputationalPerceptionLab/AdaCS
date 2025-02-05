import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.adaptive_network import AdaCS
from torch import nn
import os
import argparse
from tqdm import tqdm
from collections.abc import Iterable
from data_utils import TrainDatasetFromFolder
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Train code for AdaCS_TPAMI_2024 by Chenxi Qiu')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--block_size', default=8, type=int, help='CS block size')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--learning_rate', default=0.0003, type=float, help='learning rate')
parser.add_argument('--save_freqency', default=100, type=int, help='save models frequency')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--batchSize', default=16, type=int, help='train batch size')
parser.add_argument('--start_stage', default=0, type=int, help='training start stage')
parser.add_argument('--total_stage', default=4, type=int, help='total stage')
parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--model_name', default='./weights/AdaCS', type=str, help='save path')
parser.add_argument('--generatorWeights', type=str, default='', help="path to pretrained weights (to continue training)")
opt = parser.parse_args()

LOAD_EPOCH = 0

for k, v in sorted(vars(opt).items()):
    print(k, '=', v)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_list

train_set = TrainDatasetFromFolder('your traning data path', crop_size=opt.crop_size, blocksize=opt.block_size) # /data/qiuchenxi/JPEGImages
train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=opt.batchSize, shuffle=True)
l1_loss = nn.L1Loss()

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

for cur_stage in range(opt.start_stage, opt.total_stage):
    save_dir = opt.model_name + '/stage' + str(cur_stage)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net = AdaCS(stage=cur_stage, blocksize=opt.block_size)
    # resume training
    if opt.generatorWeights != '' and cur_stage == opt.start_stage:
        start_epoch = opt.loadEpoch
        pretrained_dict = torch.load(opt.generatorWeights)
        #
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        opt.learning_rate = 0.0003
        print('resume train')
        set_freeze_by_names(net, 'cs%d' % 0)
        print('freeze cs0')
    elif cur_stage > 0:
        start_epoch = 0
        LOAD_EPOCH = 0
        opt.learning_rate = 0.0003
        former_layers_path = opt.model_name + '/stage' + str(cur_stage-1)
        pretrained_dict = torch.load(os.path.join(former_layers_path, 'net_epoch_{}.pth'.format(opt.num_epochs)))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print('train stage %s' % str(cur_stage))
    else:
        start_epoch = 0
    if torch.cuda.is_available():
        net.cuda()
        l1_loss.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 240, 280, 290], gamma=0.5)
    for i in range(cur_stage):
        set_freeze_by_names(net, 'cs%d' % i)
        set_freeze_by_names(net, 'recon%d' % i)

    best_loss = float('inf')
    for epoch in range(start_epoch, opt.num_epochs + 1):
        train_bar = tqdm(train_loader, ncols=140)
        running_results = {'batch_sizes': 0, 'pixel_loss': 0, 'RIP_loss': 0, }

        net.train()
        scheduler.step()

        for data, target in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            gt_img = Variable(target)
            if torch.cuda.is_available():
                gt_img = gt_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            res, phi_phiT = net(z)

            optimizer.zero_grad()
            I_matrix = torch.eye(phi_phiT.shape[0]).cuda()
            gamma = torch.Tensor([0.0001]).cuda()
            loss_orth = l1_loss(phi_phiT, I_matrix)
            pixel_loss = l1_loss(res, gt_img)
            loss = pixel_loss + torch.mul(gamma, loss_orth)
            loss.backward()
            optimizer.step()
            running_results['pixel_loss'] += pixel_loss.item() * batch_size
            running_results['RIP_loss'] += loss_orth.item() * batch_size

            train_bar.set_description(desc='[%d] Loss_pixel: %.4f Loss_RIP: %.4f lr: %.7f' % (
                epoch,
                running_results['pixel_loss'] / running_results['batch_sizes'],
                running_results['RIP_loss'] / running_results['batch_sizes'],
                optimizer.param_groups[0]['lr']))
    # for saving model
        if epoch % opt.save_freqency == 0:
            torch.save(net.state_dict(), save_dir + '/net_epoch_%d.pth' % (epoch))
        if running_results['pixel_loss'] < best_loss:
            best_loss = running_results['pixel_loss']
            torch.save(net.state_dict(), save_dir + '/net_epoch_best.pth')

