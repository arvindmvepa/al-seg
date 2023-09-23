import sys
from active_learning.model.wsl4mis_model import WSL4MISModel, DeepBayesianWSL4MISMixin
import json
import os
import logging
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from wsl4mis.code.dataloaders.dataset import BaseDataSets
from wsl4mis.code.dataloaders.dataset_s2l import BaseDataSets_s2l, RandomGenerator_s2l
from wsl4mis.code.networks.net_factory import net_factory
from wsl4mis.code.utils import losses
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch
from wsl4mis.code.val_2D import test_single_volume



class DMPLSS2LModel(WSL4MISModel):
    """DMPLS Scribble2Label Model class"""
    
    def __init__(self, ann_type="scribble", data_root="wsl4mis_data/ACDC", ensemble_size=1,
                 seg_model='unet', num_classes=4, batch_size=6, base_lr=0.01, max_iterations=60000, 
                 deterministic=1, patch_size=(256, 256), thr_iter=6000, thr_conf=0.8, period_iter=100, 
                 alpha=0.2, seed=0, gpus="0", tag=""):
        super().__init__(ann_type=ann_type, data_root=data_root, ensemble_size=ensemble_size, seed=seed, gpus=gpus,
                         tag=tag)
        self.seg_model = seg_model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.deterministic = deterministic
        self.base_lr = base_lr
        self.patch_size = patch_size
        self.thr_iter = thr_iter
        self.thr_conf = thr_conf
        self.period_iter = period_iter
        self.alpha = alpha
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
        logging.info(str(self.__dict__))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        model = net_factory(net_type=self.seg_model, in_chns=1, class_num=self.num_classes)
        model = model.to(self.gpus)

        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        db_train = BaseDataSets_s2l(split="train", train_file=train_file, data_root=self.data_root, 
                                    transform=transforms.Compose([RandomGenerator_s2l(self.patch_size)]))
        db_val = BaseDataSets(split="val", val_file=self.orig_val_im_list_file, data_root=self.data_root)

        trainloader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.base_lr,
                              momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss(ignore_index=4)
        u_ce_loss = CrossEntropyLoss(ignore_index=4)

        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num = 0
        max_epoch = self.max_iter(cur_total_oracle_split, cur_total_pseudo_split) // len(trainloader) + 1
        best_performance = -1.0
        iterator = tqdm(range(max_epoch), ncols=70)
        alpha = 1.0

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch, weight_batch = sampled_batch['image'], sampled_batch['scribble'], sampled_batch['weight']
                volume_batch, label_batch = volume_batch.to(self.gpus), label_batch.to(self.gpus)

                sys.stdout.flush()

                outputs = model(volume_batch)
                loss_ce = ce_loss(outputs, label_batch.long())
                if iter_num < self.thr_iter:
                    loss = loss_ce
                else:
                    scribbles = label_batch.long().cpu()
                    mean_0, mean_1, mean_2, mean_3 = weight_batch[..., 0], weight_batch[..., 1], weight_batch[..., 2], \
                        weight_batch[..., 3]

                    u_labels_0 = torch.where((mean_0 > self.thr_conf) & (scribbles == 4),
                                                torch.zeros_like(mean_0), 4. * torch.ones_like(scribbles)).to(self.gpus)
                    u_labels_1 = torch.where((mean_1 > self.thr_conf) & (scribbles == 4),
                                                torch.zeros_like(mean_1) + 1, 4. * torch.ones_like(scribbles)).to(self.gpus)
                    u_labels_2 = torch.where((mean_2 > self.thr_conf) & (scribbles == 4),
                                                torch.zeros_like(mean_2) + 2, 4. * torch.ones_like(scribbles)).to(self.gpus)
                    u_labels_3 = torch.where((mean_3 > self.thr_conf) & (scribbles == 4),
                                                torch.zeros_like(mean_3) + 3, 4. * torch.ones_like(scribbles)).to(self.gpus)
                    u_labels = torch.ones_like(u_labels_0).long() * 4
                    u_labels[u_labels_0 == 0] = 0
                    u_labels[u_labels_1 == 1] = 1
                    u_labels[u_labels_2 == 2] = 2
                    u_labels[u_labels_3 == 3] = 3
                    loss_u = u_ce_loss(outputs, u_labels)
                    loss = loss_ce + 0.5 * loss_u

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = self.base_lr * (
                            1.0 - iter_num / self.max_iter(cur_total_oracle_split, cur_total_pseudo_split)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, alpha: %f' %
                    (iter_num, loss.item(), loss_ce.item(), alpha))

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
                    
                if iter_num > 0 and iter_num % self.period_iter == 0:
                    logging.info("Update weight start!")
                    ds = trainloader.dataset
                    for idx, images in ds.images.items():
                        img = images['image']
                        img = zoom(
                            img, (256 / img.shape[0], 256 / img.shape[1]), order=0)
                        img = torch.from_numpy(img).unsqueeze(
                            0).unsqueeze(0).to(self.gpus)
                        with torch.no_grad():
                            pred = torch.nn.functional.softmax(model(img), dim=1)
                        pred = pred.squeeze(0).cpu().numpy()
                        pred = zoom(
                            pred, (1, images['image'].shape[0] / 256, images['image'].shape[1] / 256), order=0)
                        pred = torch.from_numpy(pred)
                        weight = torch.from_numpy(images['weight'])
                        x0, x1, x2, x3 = pred[0], pred[1], pred[2], pred[3]
                        weight[..., 0] = self.alpha * x0 + \
                            (1 - self.alpha) * weight[..., 0]
                        weight[..., 1] = self.alpha * x1 + \
                            (1 - self.alpha) * weight[..., 1]
                        weight[..., 2] = self.alpha * x2 + \
                            (1 - self.alpha) * weight[..., 2]
                        weight[..., 3] = self.alpha * x3 + \
                            (1 - self.alpha) * weight[..., 3]
                        trainloader.dataset.images[idx]['weight'] = weight.numpy()

                    logging.info("update weight end")

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
    
    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        full_db_train = BaseDataSets_s2l(split="train", train_file=self.orig_train_im_list_file, data_root=self.data_root, 
                                    transform=transforms.Compose([RandomGenerator_s2l(self.patch_size)]))
        full_trainloader = DataLoader(full_db_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        ann_db_train = BaseDataSets_s2l(split="train", train_file=train_file, data_root=self.data_root, 
                                    transform=transforms.Compose([RandomGenerator_s2l(self.patch_size)]))
        train_preds = {}
        for i_batch, sampled_batch in tqdm(enumerate(full_trainloader)):
            volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['scribble'], sampled_batch['idx']
            volume_batch, label_batch, idx = volume_batch.to(self.gpus), label_batch.to(self.gpus), idx.cpu()[0]
            # skip images that are already annotated
            if full_db_train.sample_list[idx] in ann_db_train.sample_list:
                continue
            slice_basename = os.path.basename(full_db_train.sample_list[idx])
            outputs = model(volume_batch)
            outputs_soft = self.extract_model_prediction(outputs)
            train_preds[slice_basename] = np.float16(outputs_soft.cpu().detach().numpy())
        train_preds_path = os.path.join(snapshot_dir, "train_preds.npz")
        np.savez_compressed(train_preds_path, **train_preds)

    def _extract_model_prediction_channel(self, outputs):
        return outputs
    
    def _test_single_volume(self):
        return test_single_volume

    @property
    def model_string(self):
        return "dmpls_s2l"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DMPLSS2LModel"
        return json.dumps(mapping)


class DeepBayesianDMPLSS2LModel(DeepBayesianWSL4MISMixin, DMPLSS2LModel):

    def __init__(self, T=40, db_score_func="mean_probs", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T
        self.db_score_func = db_score_func
        
    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.train()
        full_db_train = BaseDataSets_s2l(split="train", train_file=self.orig_train_im_list_file, data_root=self.data_root, 
                                    transform=transforms.Compose([RandomGenerator_s2l(self.patch_size)]))
        full_trainloader = DataLoader(full_db_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        ann_db_train = BaseDataSets_s2l(split="train", train_file=train_file, data_root=self.data_root, 
                                    transform=transforms.Compose([RandomGenerator_s2l(self.patch_size)]))
        train_preds = {}

        print("Start Monte Carlo dropout forward passes on the inferences!")
        print("Each inference will be repeated {} times.".format(self.T))
        with torch.no_grad():  # All computations inside this context will not track gradients
            for i_batch, sampled_batch in tqdm(enumerate(full_trainloader)):
                volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['scribble'], sampled_batch['idx']
                volume_batch, label_batch, idx = volume_batch.to(self.gpus), label_batch.to(self.gpus), idx.cpu()[0]

                # skip images that are already annotated
                if full_db_train.sample_list[idx] in ann_db_train.sample_list:
                    continue

                slice_basename = os.path.basename(full_db_train.sample_list[idx])

                # Use repeat_interleave to create a batch with the same volume repeated T times
                volume_batch_repeated = volume_batch.repeat_interleave(self.T, dim=0)

                # Use the model to get the repeated outputs
                outputs = model(volume_batch_repeated)
                outputs = self.extract_model_prediction(outputs, batch_size=self.T)
                db_scores = self.get_db_score(outputs)
                train_preds[slice_basename] = np.float32(db_scores.cpu().detach().numpy())

        train_preds_path = os.path.join(snapshot_dir, "train_preds.npz")
        np.savez_compressed(train_preds_path, **train_preds)
    

    @property
    def model_string(self):
        return "db_dmpls_s2l"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "DeepBayesianDMPLSS2LModel"
        return json.dumps(mapping)