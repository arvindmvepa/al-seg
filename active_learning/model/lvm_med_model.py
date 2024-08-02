from abc import abstractmethod
from active_learning.model.base_model import BaseModel, SoftmaxMixin
from active_learning.model.db_scoring_functions import db_scoring_functions
import json
import os
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from wsl4mis.code.dataloaders.dataset import BaseDataSets, RandomGenerator
from lvm_med.utils.endtoend import dice_loss
import logging
import sys
from lvm_med.evaluate import evaluate, evaluate_3d_iou
import torch
import h5py
import segmentation_models_pytorch as smp


class LVMMedModel(SoftmaxMixin, BaseModel):
    """WSL4MIS Model class"""

    def __init__(self, dataset="ACDC", ann_type="scribble", ensemble_size=1, in_chns=3, num_epochs=50,
                 base_original_checkpoint="scratch", batch_size=8, base_lr=0.0001, train_beta1=0.9, train_beta2=0.999,
                 train_weight_decay=0, train_scheduler=0, patch_size=(256, 256), amp=False, inf_train_type="preds",
                 feature_decoder_index=0, seed=0, gpus="cuda:0", tag=""):
        super().__init__(ann_type=ann_type, dataset=dataset, ensemble_size=ensemble_size, seed=seed, gpus=gpus,
                         tag=tag)
        self.in_chns = in_chns
        self.num_epochs = num_epochs
        self.base_original_checkpoint = base_original_checkpoint
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.train_beta1 = train_beta1
        self.train_beta2 = train_beta2
        self.train_weight_decay = train_weight_decay
        self.train_scheduler = train_scheduler
        self.amp = amp
        self.patch_size = patch_size
        self.inf_train_type = inf_train_type
        self.feature_decoder_index = feature_decoder_index
        self.gpus = gpus

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        logger = logging.getLogger()
        if len(logger.handlers) != 0:
            logger.handlers[0].stream.close()
            logger.handlers.clear()

        logging.basicConfig(filename=os.path.join(snapshot_dir, "log.txt"), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.info(str(self.__dict__))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        device = torch.device(self.gpus)
        if self.base_original_checkpoint == "scratch":
            net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=self.in_chns, classes=self.num_classes)
        else:
            print("Using pre-trained models from", self.base_original_checkpoint)
            net = smp.Unet(encoder_name="resnet50", encoder_weights=self.base_original_checkpoint,
                           in_channels=self.in_chns, classes=self.num_classes)
        net.to(device=device)
        optimizer = optim.Adam(net.parameters(), lr=self.base_lr, betas=(self.train_beta1, self.train_beta2),
                               eps=1e-08, weight_decay=self.train_weight_decay)
        if self.train_scheduler:
            print("Use scheduler")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-05)

        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                sup_type=self.ann_type, train_file=train_file, data_root=self.data_root)
        db_val = BaseDataSets(split="val", val_file=self.orig_val_im_list_file, data_root=self.data_root)

        trainloader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

        n_train = len(db_train)
        n_val = len(db_val)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        criterion = nn.CrossEntropyLoss()
        global_step = 0
        best_value = 0

        logging.info(f'''Starting training:
            Epochs:          {self.num_epochs}
            Train batch size:      {self.batch_size}
            Learning rate:   {self.base_lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            Patch Size:  {self.patch_size}
            Mixed Precision: {self.amp}
        ''')
        best_model_path = os.path.join(snapshot_dir, 'resenet50_best_model.pth')
        # 5. Begin training
        for epoch in range(self.num_epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
                for sampled_batch in trainloader:
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.to(self.gpus), label_batch.to(self.gpus)

                    with torch.cuda.amp.autocast(enabled=self.amp):
                        masks_pred = net(volume_batch)
                        loss = criterion(masks_pred, label_batch) + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(label_batch, self.num_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    clip_value = 1
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(volume_batch.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    """
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    """
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    if self.train_scheduler:
                        scheduler.step()
                    # Evaluation round
                    if global_step % (n_train // (1 * self.batch_size)) == 0:
                        val_dice_score, val_iou_score = evaluate(net, valloader, device, 1)
                        val_score = val_dice_score

                        if (val_score > best_value):
                            best_value = val_score
                            logging.info("New best dice score: {} at epochs {}".format(best_value, epoch + 1))
                            torch.save(net.state_dict(), best_model_path)

                        logging.info('Validation Dice score: {}, IoU score {}'.format(val_dice_score, val_iou_score))

            if epoch + 1 == self.num_epochs:
                val_dice_score, val_iou_score = evaluate(net, valloader, device, 1)
                logging.info('Validation Dice score: {}, IoU score {}'.format(val_dice_score, val_iou_score))

        logging.info("Evalutating on test set")
        logging.info("Loading best model on validation")
        return "Training Finished!"


    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        train_preds_path = os.path.join(snapshot_dir, "train_preds.npz")
        if os.path.exists(train_preds_path):
            print(f"Train preds already exist in {train_preds_path}")
            return
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        if self.inf_train_type == "features":
            model = model.encoder
        elif self.inf_train_type != "preds":
            raise ValueError(f"self.inf_train_type {self.inf_train_type } is not recognized. ust be either 'features' or 'preds'")
        model.eval()
        full_db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                     sup_type=self.ann_type, train_file=self.orig_train_im_list_file,
                                     data_root=self.data_root)
        full_trainloader = DataLoader(full_db_train, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        train_preds = {}
        for i_batch, sampled_batch in tqdm(enumerate(full_trainloader)):
            volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
            volume_batch, label_batch, idx = volume_batch.to(self.gpus), label_batch.to(self.gpus), idx.cpu()[0]
            slice_basename = os.path.basename(full_db_train.sample_list[idx])
            outputs = model(volume_batch)
            outputs_ = self.extract_train_features(outputs)
            train_preds[slice_basename] = np.float16(outputs_.cpu().detach().numpy())
        np.savez_compressed(train_preds_path, **train_preds)

    def inf_eval_model(self, eval_file, model_no, snapshot_dir, round_dir, metrics_file, cur_total_oracle_split=0,
                       cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        db_eval = BaseDataSets(split="val", val_file=eval_file, data_root=self.data_root)
        evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
        metric_list = 0.0
        results_map = {}
        for i_batch, sampled_batch in enumerate(evalloader):
            metric_i = test_single_volume_cct(sampled_batch["image"], sampled_batch["label"], model,
                                              classes=self.num_classes, gpus=self.gpus)
            metric_i = np.array(metric_i)
            results_map[sampled_batch["case"][0]] = np.mean(metric_i[:, 0])
            metric_list += metric_i
        metric_list = metric_list / len(db_eval)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        metrics = {"performance": performance, "mean_hd95": mean_hd95}
        metrics_file = os.path.join(snapshot_dir, metrics_file)
        with open(metrics_file, "w") as outfile:
            json_object = json.dumps(metrics, indent=4)
            outfile.write(json_object)
        results_map_file = os.path.join(snapshot_dir, os.path.basename(metrics_file)+"_map.json")
        with open(results_map_file, "w") as outfile:
            json_object = json.dumps(results_map, indent=4)
            outfile.write(json_object)

    def inf_val_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_eval_model(eval_file=self.orig_val_im_list_file, model_no=model_no, snapshot_dir=snapshot_dir,
                            metrics_file="val_metrics.json", round_dir=round_dir,
                            cur_total_oracle_split=cur_total_oracle_split,
                            cur_total_pseudo_split=cur_total_pseudo_split)

    def inf_test_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_eval_model(eval_file=self.orig_test_im_list_file, model_no=model_no, snapshot_dir=snapshot_dir,
                            metrics_file="test_metrics.json", round_dir=round_dir,
                            cur_total_oracle_split=cur_total_oracle_split,
                            cur_total_pseudo_split=cur_total_pseudo_split)

    def generate_model_results(self, snapshot_dir, prediction_ims=None):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        db_eval = BaseDataSets(split="val", val_file=self.orig_test_im_list_file, data_root=self.data_root)
        evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
        for i_batch, sampled_batch in enumerate(evalloader):
            case = sampled_batch["case"][0]
            case_wo_ext = os.path.basename(case).split(".")[0]
            if not any(prediction_im in sampled_batch["case"][0] for prediction_im in prediction_ims):
                continue
            prediction_i = test_single_volume_cct(sampled_batch["image"], sampled_batch["label"], model,
                                                  classes=self.num_classes, gpus=self.gpus, generate=True)
            prediction_i = np.stack(prediction_i, axis=0)
            pred_h5 = os.path.join(snapshot_dir, case_wo_ext + "_pred.h5")
            with h5py.File(pred_h5, "w") as h5f:
                h5f.create_dataset("data", data=prediction_i, dtype=np.float32)

    def load_best_model(self, snapshot_dir):
        model = net_factory(net_type=self.seg_model, in_chns=self.in_chns, class_num=self.num_classes)
        best_model_path = os.path.join(snapshot_dir, '{}_best_model.pth'.format(self.seg_model))
        model.load_state_dict(torch.load(best_model_path))
        return model

    def _extract_model_prediction_channel(self, outputs):
        return outputs[0]

    def extract_model_prediction(self, raw_model_outputs, batch_size=1):
        outputs = self._extract_model_prediction_channel(raw_model_outputs)
        outputs = torch.softmax(outputs, dim=1)
        if batch_size == 1:
            outputs = outputs[0]
        return outputs

    def extract_train_features(self, raw_model_outputs, batch_size=1):
        if self.inf_train_type == "features":
            return raw_model_outputs[self.feature_decoder_index]
        else:
            return self.extract_model_prediction(raw_model_outputs, batch_size=batch_size)

    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        new_train_im_list_file = os.path.join(round_dir,
                                              "train_al" + str(cur_total_oracle_split) + "_" + self.tag + \
                                              "_seed" + str(self.seed) \
                                                  if self.tag else \
                                                  "train_al" + str(cur_total_oracle_split) + "_seed" + str(self.seed))
        new_train_im_list_file = new_train_im_list_file + ".txt"
        return {self.file_keys[0]: new_train_im_list_file}

    def _init_train_file_info(self):
        self.orig_train_im_list_file = self.data_params['train_file']
        self.all_train_files_dict = dict()
        with open(self.orig_train_im_list_file, "r") as f:
            self.all_train_files_dict[self.file_keys[0]] = f.read().splitlines()

    def _init_val_file_info(self):
        self.orig_val_im_list_file = self.data_params["val_file"]

    def _init_test_file_info(self):
        self.orig_test_im_list_file = self.data_params["test_file"]

    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.max_iterations * (cur_total_oracle_split + cur_total_pseudo_split))

    @property
    def model_string(self):
        return "lvm_med"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "LVMMed"
        return json.dumps(mapping)

    @property
    def file_keys(self):
        return ['im']

    @property
    def im_key(self):
        return self.file_keys[0]

