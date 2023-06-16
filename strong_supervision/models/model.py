import os
import sys
import logging
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter

from strong_supervision.models.net_factory import StrongModel
from strong_supervision.dataloader.wsl4mis_dataset import WSL4MISDataset, RandomGenerator
from strong_supervision.utils import losses, val_2D, load_params


class RunExperiment(object):
    def __init__(self, exp_params_file):
        self.exp_params = load_params.load_yaml(exp_params_file)['model']

    def run(self):
        print(self.exp_params)
        self.__train_model(**self.exp_params)

    @staticmethod
    def __train_model(arch, encoder, encoder_weights, data_root, exp_dir, 
                      patch_size=(256, 256), batch_size=16, base_lr=5e-3, max_iterations=100,
                      seed=0, gpus="0", fold='fold1'):
        
        logger = logging.getLogger()
        if len(logger.handlers) != 0:
            logger.handlers[0].stream.close()
            logger.handlers.clear()

        logger_filename = os.path.join(exp_dir, "log.txt")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        logging.basicConfig(filename=logger_filename, level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
        
        model_obj = StrongModel(arch=arch, encoder_name=encoder, encoder_weights=encoder_weights)
        model = model_obj.create_model()

        if gpus == 'mps':
            model = model.to('mps')
        else:
            model = model.cuda()

        train_ds = WSL4MISDataset(split='train', transform=RandomGenerator(patch_size), data_dir=data_root)
        val_ds = WSL4MISDataset(split='val', data_dir=data_root)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0)

        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
        
        ce_loss = CrossEntropyLoss(ignore_index=4)
        dice_loss = losses.pDLoss(model_obj.num_classes, ignore_index=4)
        
        writer = SummaryWriter(os.path.join(exp_dir, 'log'))
        logging.info("{} iterations per epoch".format(len(train_loader)))

        iter_num = 0
        max_epoch = max_iterations // len(train_loader) + 1
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)

        for _ in iterator:
            for _, sampled_batch in enumerate(train_loader):
                model.train()
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                if gpus == 'mps':
                    volume_batch, label_batch = volume_batch.to('mps'), label_batch.to('mps')
                else:
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                sys.stdout.flush()

                outputs = model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
        
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss = 0.5 * (loss_ce + dice_loss(outputs_soft,
                            label_batch.unsqueeze(1)))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f' %
                    (iter_num, loss.item(), loss_ce.item()))

                if iter_num > 0 and iter_num % 200 == 0:
                    model.eval()
                    metric_list = 0.0
                    for _, sampled_batch in enumerate(val_loader):
                        metric_i = val_2D.test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], model, classes=model_obj.num_classes,
                            gpus=gpus)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(val_ds)
                    for class_i in range(model_obj.num_classes-1):
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
                        save_mode_path = os.path.join(exp_dir,
                                                    'iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance, 4)))
                        save_best = os.path.join(exp_dir + 'log.txt',
                                                '{}_best_model.pth'.format(model))
                        torch.save(model.state_dict(), save_mode_path)
                        torch.save(model.state_dict(), save_best)

                    logging.info(
                        'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    model.train()

                if iter_num % 3000 == 0:
                    save_mode_path = os.path.join(
                        exp_dir + 'log.txt', 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                iterator.close()
                break
        writer.close()
        return "Training Finished!"

