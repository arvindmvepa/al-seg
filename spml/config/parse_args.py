"""Parse CLI arguments."""

import argparse

from spml.config.default import config, update_config


def parse_args(description=''):
    """Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    # Misc parameters.
    # parser.add_argument('--snapshot_dir', required=True, type=str,
    #                     help='/path/to/snapshot/dir.')
    parser.add_argument('--snapshot_dir', type=str,
                        help='/path/to/snapshot/dir.')
    parser.add_argument('--save_dir', type=str,
                        help='/path/to/save/dir.')
    parser.add_argument('--cfg_path', required=True, type=str,
                        help='/path/to/specific/config/file.')
    parser.add_argument('--semantic_memory_dir', type=str, default=None,
                        help='/path/to/stored/memory/dir.')
    parser.add_argument('--cam_dir', type=str, default=None,
                        help='/path/to/stored/cam/dir.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='/root/dir/to/data.')
    parser.add_argument('--data_list', type=str, default=None,
                        help='/path/to/data/list.')
    # Network parameters.
    parser.add_argument('--kmeans_num_clusters', type=str,
                        help='H,W')
    parser.add_argument('--label_divisor', type=int,
                        help=2048)
    # DenseCRF parameters.
    parser.add_argument('--crf_iter_max', type=int, default=10,
                        help='number of iteration for crf.')
    parser.add_argument('--crf_pos_xy_std', type=int, default=1,
                        help='hyper paramter of crf.')
    parser.add_argument('--crf_pos_w', type=int, default=3,
                        help='hyper paramter of crf.')
    parser.add_argument('--crf_bi_xy_std', type=int, default=67,
                        help='hyper paramter of crf.')
    parser.add_argument('--crf_bi_w', type=int, default=4,
                        help='hyper paramter of crf.')
    parser.add_argument('--crf_bi_rgb_std', type=int, default=3,
                        help='hyper paramter of crf.')

    args, rest = parser.parse_known_args()

    # Update config with arguments.
    update_config(args.cfg_path)

    args = parser.parse_args()

    return args


def parse_option_SimCLR():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='4,8,12',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt, unknown = parser.parse_known_args()

    # set the path according to the environment
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True
    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     if opt.cosine:
    #         eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #         opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #                 1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    #     else:
    #         opt.warmup_to = opt.learning_rate

    return opt