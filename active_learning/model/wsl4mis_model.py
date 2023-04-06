import sys
sys.path.append("./wsl4mis/code")
from active_learning.model.base_model import BaseModel
import json
import os
import logging
import numpy as np
from tqdm import tqdm
from wsl4mis.code.networks.net_factory import net_factory
from wsl4mis.code.dataloaders.dataset import BaseDataSets, RandomGenerator
from wsl4mis.code.utils import losses
from wsl4mis.code.val_2D import test_single_volume_cct
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
import torch


class DMPLSModel(BaseModel):
    """DMPLS Model class"""

    def __init__(self, ann_type="scribble", data_root="/home/asjchoi/WSL4MIS/data/ACDC", ensemble_size=1,
                 seg_model='unet_cct', num_classes=4, batch_size=6, base_lr=0.01, max_iterations=60000, deterministic=1,
                 patch_size=(256,256), seed=0, gpus="0", tag="",
                 virtualenv='/home/asjchoi/WSL4MIS/wsl4mis-env'):
        super().__init__(ann_type=ann_type, data_root=data_root, ensemble_size=ensemble_size, seed=seed, gpus=gpus,
                         tag=tag, virtualenv=virtualenv)
        self.seg_model = seg_model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.deterministic = deterministic
        self.base_lr = base_lr
        self.patch_size = patch_size

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0,
                    inf_train=False, save_params=None):
        if save_params is None:
            save_params = dict()

        # remove previous logger, if exists.
        # close and remove first handler, but just remove (but don't close) second handler (which is standard output)
        logger = logging.getLogger()
        if len(logger.handlers) != 0:
            logger.handlers[0].stream.close()
            logger.handlers.clear()

        logging.basicConfig(filename=os.path.join(snapshot_dir, "log.txt"), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(self.__dict__))

        model = net_factory(net_type=self.seg_model, in_chns=1, class_num=self.num_classes)

        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                sup_type=self.ann_type, train_file=train_file, data_root=self.data_root)
        db_val = BaseDataSets(split="val", val_file=self.orig_val_im_list_file, data_root=self.data_root)

        trainloader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.base_lr,
                              momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss(ignore_index=4)
        dice_loss = losses.pDLoss(self.num_classes, ignore_index=4)

        writer = SummaryWriter(os.path.join(snapshot_dir, 'log'))
        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num = 0
        max_epoch = self.max_iter(cur_total_oracle_split, cur_total_pseudo_split) // len(trainloader) + 1
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)
        alpha = 1.0

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                sys.stdout.flush()

                outputs, outputs_aux1 = model(
                    volume_batch)
                outputs_soft1 = torch.softmax(outputs, dim=1)
                outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

                loss_ce1 = ce_loss(outputs, label_batch[:].long())
                loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
                loss_ce = 0.5 * (loss_ce1 + loss_ce2)

                beta = self.random_gen.random() + 1e-10

                pseudo_supervision = torch.argmax(
                    (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)

                loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))

                loss = loss_ce + 0.5 * loss_pse_sup
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = self.base_lr * (1.0 - iter_num / self.max_iter(cur_total_oracle_split, cur_total_pseudo_split)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, alpha: %f' %
                    (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), alpha))

                if iter_num % 20 == 0:
                    image = volume_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(
                        outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction',
                                     outputs[0, ...] * 50, iter_num)
                    labs = label_batch[0, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)

                if iter_num > 0 and iter_num % 200 == 0:
                    model.eval()
                    metric_list = 0.0
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume_cct(
                            sampled_batch["image"], sampled_batch["label"], model, classes=self.num_classes)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    for class_i in range(self.num_classes-1):
                        writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                          metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                          metric_list[class_i, 1], iter_num)

                    performance = np.mean(metric_list, axis=0)[0]

                    mean_hd95 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('info/val_mean_dice', performance, iter_num)
                    writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                    if performance > best_performance:
                        best_performance = performance
                        save_mode_path = os.path.join(snapshot_dir,
                                                      'iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance, 4)))
                        save_best = os.path.join(snapshot_dir,
                                                 '{}_best_model.pth'.format(self.seg_model))
                        torch.save(model.state_dict(), save_mode_path)
                        torch.save(model.state_dict(), save_best)

                    logging.info(
                        'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    model.train()

                if iter_num > 0 and iter_num % 500 == 0:
                    if alpha > 0.01:
                        alpha = alpha - 0.01
                    else:
                        alpha = 0.01

                if iter_num % 3000 == 0:
                    save_mode_path = os.path.join(
                        snapshot_dir, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num >= self.max_iter(cur_total_oracle_split, cur_total_pseudo_split):
                    break
            if iter_num >= self.max_iter(cur_total_oracle_split, cur_total_pseudo_split):
                iterator.close()
                break
        writer.close()
        return "Training Finished!"

    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split, **kwargs):
        new_train_im_list_file = os.path.join(round_dir,
                                              "train_al" + str(cur_total_oracle_split) + "_" + self.tag + \
                                              "_seed" + str(self.seed) \
                                              if self.tag else \
                                              "train_al" + str(cur_total_oracle_split) + "_seed" + str(self.seed))
        new_train_im_list_file = new_train_im_list_file + ".txt"
        return {self.file_keys[0]: new_train_im_list_file}

    def _init_train_file_info(self):
        self.orig_train_im_list_file = self.model_params['train_file']
        self.all_train_files_dict = dict()
        with open(self.orig_train_im_list_file, "r") as f:
            self.all_train_files_dict[self.file_keys[0]] = f.read().splitlines()

    def _init_val_file_info(self):
        self.orig_val_im_list_file = self.model_params["val_file"]

    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.max_iterations * (cur_total_oracle_split + cur_total_pseudo_split))

    @property
    def file_keys(self):
        return ['im']

    @property
    def im_key(self):
        return self.file_keys[0]

    @property
    def model_string(self):
        return "dpmls"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DMPLSModel"
        return json.dumps(mapping)
