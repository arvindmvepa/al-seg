"""Script for training pixel-wise embeddings by pixel-segment
contrastive learning loss.
"""

from __future__ import print_function, division
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.parallel.scatter_gather as scatter_gather
import tensorboardX
from tqdm import tqdm

from lib.nn.parallel.data_parallel import DataParallel
# from lib.nn.optimizer import SGD
from lib.nn.sync_batchnorm.batchnorm import convert_model
from lib.nn.sync_batchnorm.replicate import patch_replication_callback
from spml.config.default import config
from spml.config.parse_args import parse_args, parse_option_SimCLR
import spml.utils.general.train as train_utils
import spml.utils.general.vis as vis_utils
import spml.utils.general.others as other_utils
import spml.models.utils as model_utils
from spml.data.datasets.list_tag_dataset import ListDatasetSimCLR
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
# from spml.models.predictions.segsort import segsort
from spml.models.predictions.segsort_softmax import segsort
from spml.models.predictions.softmax_classifier import softmax_classifier

import time
import torch.optim as optim
from torchvision import transforms
from SupContrast.util import TwoCropTransform, AverageMeter
from SupContrast.util import adjust_learning_rate, warmup_learning_rate
from SupContrast.util import set_optimizer, save_model
from SupContrast.networks.resnet_big import SupConResNet
from SupContrast.losses import SupConLoss

from spml.models.simclr import SimCLRResNet

torch.cuda.manual_seed_all(235)
torch.manual_seed(235)

cudnn.enabled = True
cudnn.benchmark = True


def inference(loader, model, criterion, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, images in enumerate(loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        # compute loss
        features = model(images)
        bsz = int(features.shape[0] / 2)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    """Training for pixel-wise embeddings by pixel-segment
    contrastive learning loss.
    """
    # Retreve experiment configurations.
    args = parse_args('Training for pixel-wise embeddings.')
    opt = parse_option_SimCLR()

    # Retrieve GPU informations.
    device_ids = [int(i) for i in config.gpus.split(',')]
    gpu_ids = [torch.device('cuda', i) for i in device_ids]
    num_gpus = len(gpu_ids)

    normalize = transforms.Normalize(mean=config.network.pixel_means, std=config.network.pixel_stds)

    img_size = int(512*config.network.img_scale)

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.ToTensor(),
        normalize,
    ])

    train_transform = TwoCropTransform(train_transform)

    # Create data loaders.
    dataset = ListDatasetSimCLR(
        data_dir=args.data_dir,
        data_list=args.data_list,
        img_scale=config.network.img_scale,
        transform=train_transform,
        img_mean=config.network.pixel_means,
        img_std=config.network.pixel_stds,
        size=config.train.crop_size,
        random_crop=config.train.random_crop,
        random_scale=config.train.random_scale,
        random_mirror=config.train.random_mirror,
        training=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=num_gpus * config.num_threads,
        drop_last=False)
        # collate_fn=train_dataset.collate_fn)

    embedding_model = resnet_101_deeplab(config).cuda()

    pretrained = 0
    if config.network.pretrained == "0":
        print('Training from scratch')
        pretrained = 0
    else:
        print('Loading pre-trained model: {:s}'.format(config.network.pretrained))
        embedding_model.load_state_dict(torch.load(config.network.pretrained))
        pretrained = 1

    # Add feature head for ResNet backbone
    embedding_model = SimCLRResNet(embedding_model, img_dim=img_size)

    # Use synchronize batchnorm.
    if config.network.use_syncbn:
        embedding_model = convert_model(embedding_model).cuda()

    criterion = SupConLoss(temperature=opt.temp).cuda()

    # Distribute model weights to multi-gpus.
    embedding_model.encoder = DataParallel(embedding_model.encoder,
                                           device_ids=device_ids,
                                           gather_output=False)

    if config.network.use_syncbn:
        patch_replication_callback(embedding_model.encoder)

    # training routine
    for epoch in range(opt.epochs):

        # train for one epoch
        time1 = time.time()
        loss = inference(loader, embedding_model, criterion, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

if __name__ == '__main__':
    main()
