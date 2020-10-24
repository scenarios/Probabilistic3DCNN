from philly_distributed_utils.env import get_master_ip
from philly_distributed_utils.distributed import ompi_size

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of NAS_spatiotemporal")
parser.add_argument('--dataset', type=str, default="something")
parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="/mnt/data/")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="Dense3D121")
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

parser.add_argument('--enable_nasas', default=False, action="store_true",
                    help='enable NASAS for architecture search')
parser.add_argument('--temporal_nasas_only', default=False, action="store_true",
                    help='only enable NASAS on temporal axis for architecture search')

parser.add_argument('--cross_warmup', default=False, action="store_true",
                    help='cross warmup for NASAS')

parser.add_argument('--weight_reg', type=float, default=10.0,
                    help='weight regularization used for nasas')
parser.add_argument('--p_init', type=float, default=0.1,
                    help='initial p used for nasas')
parser.add_argument('--selection_mode', default=False, action="store_true",
                    help='use selection mode in nasas')
parser.add_argument('--test_mode', default=False, action="store_true",
                    help='use test mode in nasas')
parser.add_argument('--finetune_mode', default=False, action="store_true",
                    help='use finetune mode in nasas')
parser.add_argument('--training_size', default=86017, type=int,
                    help='number of training samples')


parser.add_argument('--net_version', default='pure_fused', type=str,
                    help='densenet 3d version')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[30, 60, 80], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")


# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--test_split', type=int, default=0,
                    help='The index of test file')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume',
                    default='/mnt/log/NAS_spatiotemporal/checkpoint/warmup/NAS_sptp_something_RGB_Dense3D121_avg_segment1_e50_droprate0.5_num_dense_sample32_dense/ckpt.best.pth.tar',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--break_resume', default=False, action="store_true",
                    help='if do break restore')
parser.add_argument('--warmup', default=False, action="store_true",
                    help='if do warmup initialization')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='/mnt/log/NAS_spatiotemporal/log')
parser.add_argument('--root_model', type=str, default='/mnt/log/NAS_spatiotemporal/checkpoint')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--dense_sample_stride', default=1, type=int, help='dense sample stride for dense sample')
parser.add_argument('--num_dense_sample', default=32, type=int, help='dense sample number for dense sample')
parser.add_argument('--random_dense_sample_stride', default=False, action="store_true", help='use random dense sample stride for video dataset')

parser.add_argument('--syncbn', default=False, action="store_true", help='Synchronized batch normalization')
parser.add_argument('--use_zip', default=False, action="store_true", help='Use ZIP file for data I/O')
parser.add_argument('--freeze_bn', default=False, action="store_true", help='Freeze batch normalization')
# ========================= Distributed Configs ==========================
parser.add_argument('--local_rank', type=int)
parser.add_argument('--node_rank', type=int, default=-1)
parser.add_argument('--dist-url',
                    default='', #'tcp://' + get_master_ip() + ':23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world-size', default=0,#ompi_size(),
                    type=int, help='number of distributed processes')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--philly-mpi-multi-node', default=False,action="store_true",
                    help='nccl multiple node distributed')
parser.add_argument('--philly-nccl-multi-node', default=False,action="store_true",
                    help='nccl multiple node distributed')
