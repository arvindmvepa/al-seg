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

torch.cuda.manual_seed_all(235)
torch.manual_seed(235)

cudnn.enabled = True
cudnn.benchmark = True


class SimCLRResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, encoder, img_dim, feat_dim=128):
        super(SimCLRResNet, self).__init__()
        self.encoder = encoder

        example = torch.zeros((1, 3, img_dim, img_dim)).cuda()
        encoder_out_shape = self.encoder({'image': example})['embedding'].shape
        dim_in = encoder_out_shape[1] * encoder_out_shape[2] * encoder_out_shape[3]


        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            # nn.Linear(dim_in, feat_dim)
        )

    def forward(self, x):
        data = {'image': x}
        feat = self.encoder(data)[0]['embedding']
        feat = torch.flatten(feat, start_dim=1)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    # for idx, (images, labels) in enumerate(train_loader):
    for idx, images in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        bsz = int(features.shape[0] / 2)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
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

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = TwoCropTransform(train_transform)

    # Create data loaders.
    train_dataset = ListDatasetSimCLR(
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
    embedding_model = SimCLRResNet(embedding_model, img_dim=int(512*config.network.img_scale))

    # Use synchronize batchnorm.
    if config.network.use_syncbn:
        embedding_model = convert_model(embedding_model).cuda()

    # Use customized optimizer and pass lr=1 to support different lr for
    # different weights.
    # print(type(embedding_model.encoder.get_params_lr()[0]))
    # print(type(list(embedding_model.head.parameters()))[0])

    # TODO: add head parameters properly
    # optimizer = SGD(
    #     embedding_model.encoder.get_params_lr(),
    #     # embedding_model.encoder.get_params_lr() + list(embedding_model.head.parameters()),
    #     lr=1,
    #     momentum=config.train.momentum,
    #     weight_decay=config.train.weight_decay)
    optimizer = optim.SGD(
        embedding_model.parameters(),
        # embedding_model.encoder.get_params_lr() + list(embedding_model.head.parameters()),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay)
    optimizer.zero_grad()

    criterion = SupConLoss(temperature=opt.temp).cuda()

    # Distribute model weights to multi-gpus.
    embedding_model.encoder = DataParallel(embedding_model.encoder,
                                           device_ids=device_ids,
                                           gather_output=False)

    if config.network.use_syncbn:
        patch_replication_callback(embedding_model.encoder)

    # curr_iter = 0

    # training routine
    # pbar = tqdm(range(config.train.max_iteration))
    for epoch in range(opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, embedding_model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    torch.save(embedding_model.encoder.module.state_dict(),
               "/home/asjchoi/SPML/snapshots/imagenet/trained/simclr_resnet101_pretrained{}.pth".format(pretrained))

    #
    # # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)




    # # Create memory bank.
    # memory_banks = {}
    #
    # # start training
    # curr_iter = config.train.begin_iteration
    # train_iterator = train_loader.__iter__()
    # iterator_index = 0
    # pbar = tqdm(range(curr_iter, config.train.max_iteration))
    # for curr_iter in pbar:
    #     # Check if the rest of datas is enough to iterate through;
    #     # otherwise, re-initiate the data iterator.
    #     if iterator_index + num_gpus >= len(train_loader):
    #         train_iterator = train_loader.__iter__()
    #         iterator_index = 0
    #
    #     # Feed-forward.
    #     image_batch, label_batch = other_utils.prepare_datas_and_labels_mgpu(
    #         train_iterator, gpu_ids)
    #     iterator_index += num_gpus
    #
    #     # Generate embeddings, clustering and prototypes.
    #     embeddings = embedding_model(*zip(image_batch, label_batch))
    #     print(embeddings[1]['embedding'].shape)
    #     exit(0)
    #
    #     # Synchronize cluster indices and computer prototypes.
    #     c_inds = [emb['cluster_index'] for emb in embeddings]
    #     cb_inds = [emb['cluster_batch_index'] for emb in embeddings]
    #     cs_labs = [emb['cluster_semantic_label'] for emb in embeddings]
    #     ci_labs = [emb['cluster_instance_label'] for emb in embeddings]
    #     c_embs = [emb['cluster_embedding'] for emb in embeddings]
    #     c_embs_with_loc = [emb['cluster_embedding_with_loc']
    #                        for emb in embeddings]
    #     (prototypes, prototypes_with_loc,
    #      prototype_semantic_labels, prototype_instance_labels,
    #      prototype_batch_indices, cluster_indices) = (
    #         model_utils.gather_clustering_and_update_prototypes(
    #             c_embs, c_embs_with_loc,
    #             c_inds, cb_inds,
    #             cs_labs, ci_labs,
    #             'cuda:{:d}'.format(num_gpus - 1)))
    #
    #     for i in range(len(label_batch)):
    #         label_batch[i]['prototype'] = prototypes[i]
    #         label_batch[i]['prototype_with_loc'] = prototypes_with_loc[i]
    #         label_batch[i]['prototype_semantic_label'] = prototype_semantic_labels[i]
    #         label_batch[i]['prototype_instance_label'] = prototype_instance_labels[i]
    #         label_batch[i]['prototype_batch_index'] = prototype_batch_indices[i]
    #         embeddings[i]['cluster_index'] = cluster_indices[i]
    #
    #     semantic_tags = model_utils.gather_and_update_datas(
    #         [lab['semantic_tag'] for lab in label_batch],
    #         'cuda:{:d}'.format(num_gpus - 1))
    #     for i in range(len(label_batch)):
    #         label_batch[i]['semantic_tag'] = semantic_tags[i]
    #         label_batch[i]['prototype_semantic_tag'] = torch.index_select(
    #             semantic_tags[i],
    #             0,
    #             label_batch[i]['prototype_batch_index'])
    #
    #     # Add memory bank to label batch.
    #     for k in memory_banks.keys():
    #         for i in range(len(label_batch)):
    #             assert (label_batch[i].get(k, None) is None)
    #             label_batch[i][k] = [m.to(gpu_ids[i]) for m in memory_banks[k]]
    #
    #     # Compute loss.
    #     outputs = prediction_model(*zip(embeddings, label_batch))
    #     outputs = scatter_gather.gather(outputs, gpu_ids[0])
    #     losses = []
    #     for k in ['sem_ann_loss', 'sem_occ_loss', 'img_sim_loss', 'feat_aff_loss']:
    #         loss = outputs.get(k, None)
    #         if loss is not None:
    #             outputs[k] = loss.mean()
    #             losses.append(outputs[k])
    #     loss = sum(losses)
    #     acc = outputs['accuracy'].mean()
    #
    #     # Write to tensorboard summary.
    #     writer = (summary_writer if curr_iter % config.train.tensorboard_step == 0
    #               else None)
    #     if writer is not None:
    #         summary_vis = []
    #         summary_val = {}
    #         # Gather labels to cpu.
    #         cpu_label_batch = scatter_gather.gather(label_batch, -1)
    #         summary_vis.append(vis_utils.convert_label_to_color(
    #             cpu_label_batch['semantic_label'], color_map))
    #         summary_vis.append(vis_utils.convert_label_to_color(
    #             cpu_label_batch['instance_label'], color_map))
    #
    #         # Gather outputs to cpu.
    #         vis_names = ['embedding']
    #         cpu_embeddings = scatter_gather.gather(
    #             [{k: emb.get(k, None) for k in vis_names} for emb in embeddings],
    #             -1)
    #         for vis_name in vis_names:
    #             if cpu_embeddings.get(vis_name, None) is not None:
    #                 summary_vis.append(vis_utils.embedding_to_rgb(
    #                     cpu_embeddings[vis_name], 'pca'))
    #
    #         val_names = ['sem_ann_loss', 'sem_occ_loss',
    #                      'img_sim_loss', 'feat_aff_loss',
    #                      'accuracy']
    #         for val_name in val_names:
    #             if outputs.get(val_name, None) is not None:
    #                 summary_val[val_name] = outputs[val_name].mean().to('cpu')
    #
    #         vis_utils.write_image_to_tensorboard(summary_writer,
    #                                              summary_vis,
    #                                              summary_vis[-1].shape[-2:],
    #                                              curr_iter)
    #         vis_utils.write_scalars_to_tensorboard(summary_writer,
    #                                                summary_val,
    #                                                curr_iter)
    #
    #     # Backward propogation.
    #     if config.train.lr_policy == 'step':
    #         lr = train_utils.lr_step(config.train.base_lr,
    #                                  curr_iter,
    #                                  config.train.decay_iterations,
    #                                  config.train.warmup_iteration)
    #     else:
    #         lr = train_utils.lr_poly(config.train.base_lr,
    #                                  curr_iter,
    #                                  config.train.max_iteration,
    #                                  config.train.warmup_iteration)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step(lr)
    #
    #     # Update memory banks.
    #     with torch.no_grad():
    #         for k in label_batch[0].keys():
    #             if 'prototype' in k and 'memory' not in k:
    #                 memory = label_batch[0][k].clone().detach()
    #                 memory_key = 'memory_' + k
    #                 if memory_key not in memory_banks.keys():
    #                     memory_banks[memory_key] = []
    #                 memory_banks[memory_key].append(memory)
    #                 if len(memory_banks[memory_key]) > config.train.memory_bank_size:
    #                     memory_banks[memory_key] = memory_banks[memory_key][1:]
    #
    #         # Update batch labels.
    #         for k in ['memory_prototype_batch_index']:
    #             memory_labels = memory_banks.get(k, None)
    #             if memory_labels is not None:
    #                 for i, memory_label in enumerate(memory_labels):
    #                     memory_labels[i] += config.train.batch_size * num_gpus
    #
    #     # Snapshot the trained model.
    #     if ((curr_iter + 1) % config.train.snapshot_step == 0
    #             or curr_iter == config.train.max_iteration - 1):
    #         model_state_dict = {
    #             'embedding_model': embedding_model.module.state_dict(),
    #             'prediction_model': prediction_model.module.state_dict()}
    #         torch.save(model_state_dict,
    #                    model_path_template.format(curr_iter))
    #         torch.save(optimizer.state_dict(),
    #                    optimizer_path_template.format(curr_iter))
    #
    #     # Print loss in the progress bar.
    #     line = 'loss = {:.3f}, acc = {:.3f}, lr = {:.6f}'.format(
    #         loss.item(), acc.item(), lr)
    #     pbar.set_description(line)


if __name__ == '__main__':
    main()
