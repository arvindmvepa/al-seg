import sys
from active_learning.model.wsl4mis_model import WSL4MISModel, DeepBayesianWSL4MISMixin
import json
import os
import logging
import numpy as np
from tqdm import tqdm
from wsl4mis.code.dataloaders.dataset import BaseDataSets, RandomGenerator
from wsl4mis.code.networks.net_factory import net_factory
from wsl4mis.code.utils import losses
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch
from wsl4mis.code.val_2D import test_single_volume


class DMPLSMumfordShahLossModel(WSL4MISModel):
    """DMPLS Mumford ShahLoss Model class"""

    def __init__(self, ann_type="scribble", data_root="wsl4mis_data/ACDC", ensemble_size=1,
                 seg_model='unet', num_classes=4, batch_size=6, base_lr=0.01, max_iterations=60000, deterministic=1,
                 patch_size=(256, 256), output_dim=1, seed=0, gpus="0", tag=""):
        super().__init__(ann_type=ann_type, data_root=data_root, ensemble_size=ensemble_size, seed=seed, gpus=gpus,
                         tag=tag, output_dim=output_dim)
        self.seg_model = seg_model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.deterministic = deterministic
        self.base_lr = base_lr
        self.patch_size = patch_size
        self.gpus = gpus

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
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
        model = model.to(self.gpus)

        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                sup_type=self.ann_type, train_file=train_file, data_root=self.data_root)
        db_val = BaseDataSets(split="val", val_file=self.orig_val_im_list_file, data_root=self.data_root)

        trainloader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.base_lr,
                              momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss(ignore_index=4)
        mumfordshah_loss = losses.MumfordShah_Loss()

        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num = 0
        max_epoch = self.max_iter(cur_total_oracle_split, cur_total_pseudo_split) // len(trainloader) + 1
        best_performance = -1.0
        iterator = tqdm(range(max_epoch), ncols=70)
        alpha = 1.0

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.to(self.gpus), label_batch.to(self.gpus)

                sys.stdout.flush()

                outputs = model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_mumfordshah = mumfordshah_loss(volume_batch, outputs_soft)

                loss = loss_ce + 1e-6 * loss_mumfordshah
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = self.base_lr * (
                            1.0 - iter_num / self.max_iter(cur_total_oracle_split, cur_total_pseudo_split)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_mumfordshah: %f, alpha: %f' %
                    (iter_num, loss.item(), loss_ce.item(), loss_mumfordshah.item(), alpha))

                if iter_num > 0 and (iter_num % 200 == 0 or
                                     iter_num == self.max_iter(cur_total_oracle_split,
                                                               cur_total_pseudo_split)):
                    model.eval()
                    metric_list = 0.0
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"],
                            model, classes=self.num_classes, gpus=self.gpus)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)

                    performance = np.mean(metric_list, axis=0)[0]
                    mean_hd95 = np.mean(metric_list, axis=0)[1]

                    if performance > best_performance:
                        best_performance = performance
                        save_best = os.path.join(snapshot_dir,
                                                 '{}_best_model.pth'.format(self.seg_model))
                        torch.save(model.state_dict(), save_best)

                    logging.info(
                        'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    model.train()

                if iter_num > 0 and iter_num % (500 * (cur_total_oracle_split + cur_total_pseudo_split)) == 0:
                    if alpha > 0.01:
                        alpha = alpha - 0.01
                    else:
                        alpha = 0.01

                if iter_num >= self.max_iter(cur_total_oracle_split, cur_total_pseudo_split):
                    break
            if iter_num >= self.max_iter(cur_total_oracle_split, cur_total_pseudo_split):
                iterator.close()
                break

        return "Training Finished!"
    
    def inf_eval_model(self, eval_file, model_no, snapshot_dir, round_dir, metrics_file, cur_total_oracle_split=0,
                       cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        db_eval = BaseDataSets(split="val", val_file=eval_file, data_root=self.data_root)
        evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(evalloader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"],
                model, classes=self.num_classes, gpus=self.gpus)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_eval)

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        metrics = {"performance": performance, "mean_hd95": mean_hd95}
        metrics_file = os.path.join(snapshot_dir, metrics_file)
        with open(metrics_file, "w") as outfile:
            json_object = json.dumps(metrics, indent=4)
            outfile.write(json_object)

    @property
    def model_string(self):
        return "dmpls_mshah"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DMPLSMumfordShahLossModel"
        return json.dumps(mapping)


class DeepBayesianDMPLSMumfordShahLossModel(DeepBayesianWSL4MISMixin, DMPLSMumfordShahLossModel):

    def __init__(self, T=40, db_score_func="mean_probs", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T
        self.db_score_func = db_score_func
        self.output_dim = 1

    @property
    def model_string(self):
        return "db_dmpls_mshah"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DeepBayesianDMPLSMumfordShahLossModel"
        return json.dumps(mapping)