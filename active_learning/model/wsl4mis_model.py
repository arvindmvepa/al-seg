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
import sys


class DMPLSModel(BaseModel):
    """DMPLS Model class"""

    def __init__(self, ann_type="scribble", data_root="/home/asjchoi/WSL4MIS/data/ACDC", ensemble_size=1,
                 seg_model='unet_cct', num_classes=4, batch_size=6, base_lr=0.01, max_iterations=60000, deterministic=1,
                 patch_size=(256,256), seed=0, fold="fold1", gpus="0", tag="",
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
        self.fold = fold

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0,
                    inf_train=False, save_params=None):
        if save_params is None:
            save_params = dict()
        # set force to true to close any existing loggers
        logging.basicConfig(filename=os.path.join(snapshot_dir, "log.txt"), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', force=True)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(self.__dict__))

        model = net_factory(net_type=self.seg_model, in_chns=1, class_num=self.num_classes)
        db_train = BaseDataSets(base_dir=self.data_root, split="train", transform=transforms.Compose([
            RandomGenerator(self.patch_size)
        ]), fold=self.fold, sup_type=self.ann_type)
        db_val = BaseDataSets(base_dir=self.data_root, fold=self.fold, split="val")

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
        max_epoch = self.max_iterations // len(trainloader) + 1
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)
        alpha = 1.0

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

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

                lr_ = self.base_lr * (1.0 - iter_num / self.max_iterations) ** 0.9
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
                    image = volume_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(
                        outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction',
                                     outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
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

                if iter_num >= self.max_iterations:
                    break
            if iter_num >= self.max_iterations:
                iterator.close()
                break
        writer.close()
        return "Training Finished!"

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict, skip=False):
        print("Starting to Ensemble Predictions")
        f = open(im_score_file, "w")
        train_results_dir = os.path.join(round_dir, "*", self.model_params['train_results_dir'])
        filt_models_result_files = self._filter_unann_ims(train_results_dir, ignore_ims_dict)
        for models_result_file in tqdm(zip(*filt_models_result_files)):
            results_arr, base_name = self._convert_ensemble_results_to_arr(models_result_file)
            # calculate the score_func over the ensemble of predictions
            score = score_func(results_arr)
            f.write(f"{base_name},{np.round(score, 7)}\n")
            f.flush()
        f.close()

    def _init_val_file_info(self):
        self.val_pim_list_file = os.path.join("spml",
                                              "datasets",
                                              "voc12",
                                              self.model_params["val_pim_list"])

    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.num_epochs * self.epoch_len * (cur_total_oracle_split + cur_total_pseudo_split))

    def _set_loss_weights(self, sem_ann_concentration, sem_occ_concentration, img_sim_concentration,
                          feat_aff_concentration, sem_ann_loss_weight, sem_occ_loss_weight, word_sim_loss_weight,
                          img_sim_loss_weight, feat_aff_loss_weight):
        if sem_ann_concentration is None:
            self.sem_ann_concentration = 6
        else:
            self.sem_ann_concentration = sem_ann_concentration
        if img_sim_concentration is None:
            self.img_sim_concentration = 16
        else:
            self.img_sim_concentration = img_sim_concentration
        if feat_aff_concentration is None:
            self.feat_aff_concentration = 0
        else:
            self.feat_aff_concentration = feat_aff_concentration
        if word_sim_loss_weight is None:
            self.word_sim_loss_weight = 0
        else:
            self.word_sim_loss_weight = word_sim_loss_weight
        if img_sim_loss_weight is None:
            self.img_sim_loss_weight = 0.1
        else:
            self.img_sim_loss_weight = img_sim_loss_weight
        if feat_aff_loss_weight is None:
            self.feat_aff_loss_weight = 0
        else:
            self.feat_aff_loss_weight = feat_aff_loss_weight
        if sem_occ_concentration is None:
            if self.ann_type == "box":
                self.sem_occ_concentration = 8
            if self.ann_type == "scribble":
                self.sem_occ_concentration = 12
        else:
            self.sem_occ_concentration = sem_occ_concentration
        if sem_ann_loss_weight is None:
            if self.ann_type == "box":
                self.sem_ann_loss_weight = 0.3
            if self.ann_type == "scribble":
                self.sem_ann_loss_weight = 1.0
        else:
            self.sem_ann_loss_weight = sem_ann_loss_weight
        if sem_occ_loss_weight is None:
            if self.ann_type == "box":
                self.sem_occ_loss_weight = 0.3
            if self.ann_type == "scribble":
                self.sem_occ_loss_weight = 0.5
        else:
            self.sem_occ_loss_weight = sem_occ_loss_weight

    @property
    def model_string(self):
        return "dpmls"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DMPLSModel"
        return json.dumps(mapping)
