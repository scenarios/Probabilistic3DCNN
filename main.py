
import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset.IO import TSNDataSet
from dataset.augment import *
from args import parser
from dataset import config as dataset_config
from utils import AverageMeter, accuracy
from models.densenet_3d import densenet121
from models.resnet_3d import resnet50, resnext50_32x4d, resnext101_32x8d
from models.mobilenet_v2_3d import mobilenet_v2

from prefetch_generator import BackgroundGenerator

#from tensorboardX import SummaryWriter

from philly_distributed_utils.distributed import ompi_rank, gpu_indices

best_prec1 = 0

def arg_checker(args):
    assert args.net_version in ['v1', 'v2', 'v3', 'v4', 'v1nt', 'v2nt', 'v3nt', 'v1d2', 'v1d3', 'vt', 'pure_temporal', 'pure_spatial', 'pure_fused', 'pure_adaptive'], \
        "'v1', 'v2', 'v3', 'v4', 'v1nt', 'v2nt', 'v3nt', 'v1d2', 'v1d3', 'vt', 'pure_temporal', 'pure_spatial' are currently supported"
    if args.enable_nasas and not args.finetune_mode:
        assert args.dropout == 0.0


def train():
    global args, best_prec1, global_rank, local_rank
    args = parser.parse_args()
    arg_checker(args)

    if args.node_rank != 0:
        time.sleep(30)

    if args.philly_mpi_multi_node:
        global_rank = ompi_rank()
        local_rank = list(gpu_indices())[0]

        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=global_rank)
        print(
            "World Size is {}, Backend is {}, Init Method is {}, rank is {}".format(args.world_size, args.dist_backend,
                                                                                    args.dist_url, global_rank))
    elif args.philly_nccl_multi_node:
        torch.distributed.init_process_group(backend="nccl",
                                             init_method='env://')
        global_rank = torch.distributed.get_rank()
        local_rank = args.local_rank
    else:
        global_rank=-1
        local_rank=-1

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        torch.cuda.set_device(local_rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset+'_zip' if args.use_zip else args.dataset,
                                                                                                      args.modality,
                                                                                                      args.root_path)
    assert len([x for x in open(args.train_list)]) == args.training_size, "training size unmatched %d vs %d".format(len(args.train_list), args.training_size)

    full_arch_name = args.arch
    args.store_name = '_'.join(
        ['NAS_sptp', 'nasas' if args.enable_nasas else 'warmup', 'selection' if args.selection_mode else '', 'ls{}'.format(args.weight_reg),
         args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs), 'droprate{}'.format(args.dropout), 'num_dense_sample{}'.format(args.num_dense_sample),
         'dense_sample_stride{}'.format(args.dense_sample_stride)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.random_dense_sample_stride:
        args.store_name += '_randomstride'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        if global_rank == 0:
            check_rootfolders()
            time.sleep(5)
    else:
        check_rootfolders()
    '''
    if args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrain, num_classes=num_class, drop_rate=args.dropout)
    else:
        model = densenet121(num_classes=num_class,  pretrained=args.pretrain, drop_rate=args.dropout)
    '''
    model = eval(args.arch+'(pretrained=args.pretrain, num_classes=num_class, drop_rate=args.dropout)')
    if (args.philly_mpi_multi_node or args.philly_nccl_multi_node) and args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    input_mean = model.mean
    input_std = model.std
    #policies = model.get_optim_policies()
    train_augmentation = get_train_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True,
                                                div=(args.arch not in ['BNInception', 'InceptionV3']),
                                                roll=(args.arch in ['BNInception', 'InceptionV3']))
    val_augmentation = get_val_augmentation(div=(args.arch not in ['BNInception', 'InceptionV3']),
                                            roll=(args.arch in ['BNInception', 'InceptionV3']))

    model = model.cuda()
    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = torch.nn.parallel.DataParallel(model)

    optimizer = torch.optim.SGD(#policies,
                                model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=0.0 if args.enable_nasas else args.weight_decay )

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    dataset_train = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       #Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       #ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride,
                    num_dense_sample=args.num_dense_sample, random_dense_sample_stride=args.random_dense_sample_stride, is_zip=args.use_zip)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train) if args.philly_mpi_multi_node or args.philly_nccl_multi_node else None
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)  # prevent something not % n_GPU

    dataset_val = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       val_augmentation,
                       #GroupScale(int(scale_size)),
                       #GroupCenterCrop(crop_size),
                       #Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       #ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride, num_dense_sample=args.num_dense_sample, is_zip=args.use_zip)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val) if args.philly_mpi_multi_node or args.philly_nccl_multi_node else None
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    '''
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    '''
    if args.evaluate:
        _validate_once(val_loader, model, criterion, 0)
        return

    with open(os.path.join(args.root_log, args.store_name, 'args'+str(global_rank)+'.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = None#SummaryWriter(logdir=os.path.join(args.root_log, args.store_name))

    if args.warmup:
        assert os.path.isfile(args.resume), "No checkpoint found in %s".format(args.resume)
        print(("=> nasas warmup loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        if not args.enable_nasas or args.cross_warmup:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(args.evaluate, checkpoint['epoch'])))

    if args.break_resume:
        args.resume = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
    log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
    _validate_once(val_loader, model, criterion, log_training)
    log_training.close()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
        # train for one epoch
        _train_once(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        log_training.close()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
            prec1 = _validate_once(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            #tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            #output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            output_best = 'Best Prec@1: %.3f, Prec@1: %.3f\n' % (best_prec1, prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()
            if global_rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)
            log_training.close()


def test():
    global args, best_prec1
    args = parser.parse_args()
    arg_checker(args)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset + '_zip' if args.use_zip else args.dataset,
        args.modality,
        args.root_path)
    assert len([x for x in open(args.train_list)]) == args.training_size, "training size unmatched %d vs %d".format(
        len(args.train_list), args.training_size)

    full_arch_name = args.arch
    args.store_name = '_'.join(
        ['NAS_sptp', 'TEST',
         'ls{}'.format(args.weight_reg),
         args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs), 'droprate{}'.format(args.dropout),
         'num_dense_sample{}'.format(args.num_dense_sample),
         'dense_sample_stride{}'.format(args.dense_sample_stride)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.random_dense_sample_stride:
        args.store_name += '_randomstride'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = eval(args.arch+'(pretrained=args.pretrain, num_classes=num_class, drop_rate=args.dropout)')

    input_mean = model.mean
    input_std = model.std
    # policies = model.get_optim_policies()

    test_augmentation = get_test_augmentation(div=(args.arch not in ['BNInception', 'InceptionV3']),
                                            roll=(args.arch in ['BNInception', 'InceptionV3']))
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)
    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.test_split > 0:
        args.val_list = args.val_list.replace('val_videofolder', 'val_videofolder_split{}'.format(args.test_split))
    dataset_test = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                              test_mode=True,
                              remove_missing=True,
                              new_length=data_length,
                              modality=args.modality,
                              image_tmpl=prefix,
                              random_shift=False,
                              transform=torchvision.transforms.Compose([
                                  test_augmentation,
                                  # GroupScale(int(scale_size)),
                                  # GroupCenterCrop(crop_size),
                                  # Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                  # ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                  normalize,
                              ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride,
                              num_dense_sample=args.num_dense_sample, is_zip=args.use_zip)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    with open(os.path.join(args.root_log, args.store_name, 'args' + str(-1) + '.txt'), 'w') as f:
        f.write(str(args))

    assert os.path.isfile(args.resume), "No checkpoint found in %s".format(args.resume)
    print(("=> nasas warmup loading checkpoint '{}'".format(args.resume)))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(("=> loaded checkpoint '{}' (epoch {})"
            .format(args.evaluate, checkpoint['epoch'])))

    with open(os.path.join(args.root_log, args.store_name, 'log' + str(-1) + '.csv'), 'a') as log_test:
        _test_once(test_loader, model, criterion, epoch=-1, log=log_test)


def selection():
    global args, global_rank, local_rank
    best_prec1s = []
    worst_prec1s = []
    N_RANK=10
    args = parser.parse_args()
    arg_checker(args)

    if args.node_rank != 0:
        time.sleep(30)

    if args.philly_mpi_multi_node:
        global_rank = ompi_rank()
        local_rank = list(gpu_indices())[0]

        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=global_rank)
        print(
            "World Size is {}, Backend is {}, Init Method is {}, rank is {}".format(args.world_size, args.dist_backend,
                                                                                    args.dist_url, global_rank))
    elif args.philly_nccl_multi_node:
        torch.distributed.init_process_group(backend="nccl",
                                             init_method='env://')
        global_rank = torch.distributed.get_rank()
        local_rank = args.local_rank
    else:
        global_rank = -1
        local_rank = -1

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        torch.cuda.set_device(local_rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset+'_zip' if args.use_zip else args.dataset,
                                                                                                      args.modality,
                                                                                                      args.root_path)
    assert len([x for x in open(args.train_list)]) == args.training_size, "training size unmatched %d vs %d".format(
        len(args.train_list), args.training_size)

    full_arch_name = args.arch
    args.store_name = '_'.join(
        ['NAS_sptp', 'nasas' if args.enable_nasas else 'warmup', 'selection' if args.selection_mode else '',
         'ls{}'.format(args.weight_reg),
         args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs), 'droprate{}'.format(args.dropout),
         'num_dense_sample{}'.format(args.num_dense_sample)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        if global_rank == 0:
            check_rootfolders()
            time.sleep(5)
    else:
        check_rootfolders()

    model = densenet121(num_classes=num_class, pretrained=args.pretrain, drop_rate=args.dropout)

    input_mean = model.mean
    input_std = model.std
    # policies = model.get_optim_policies()

    val_augmentation = get_selection_augmentation(div=(args.arch not in ['BNInception', 'InceptionV3']),
                                                  roll=(args.arch in ['BNInception', 'InceptionV3']))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    dataset_val = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                             new_length=data_length,
                             modality=args.modality,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=torchvision.transforms.Compose([
                                 val_augmentation,
                                 # GroupScale(int(scale_size)),
                                 # GroupCenterCrop(crop_size),
                                 # Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                 # ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                 normalize,
                             ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride,
                             num_dense_sample=args.num_dense_sample, is_zip=args.use_zip)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val) if args.philly_mpi_multi_node or args.philly_nccl_multi_node else None
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    with open(os.path.join(args.root_log, args.store_name, 'args' + str(global_rank) + '.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = None  # SummaryWriter(logdir=os.path.join(args.root_log, args.store_name))

    for epoch in range(0, 2000):
        model = densenet121(num_classes=num_class, pretrained=args.pretrain, drop_rate=args.dropout)
        model = model.cuda()
        model = torch.nn.parallel.DataParallel(model)

        if args.warmup:
            assert os.path.isfile(args.resume), "No checkpoint found in %s".format(args.resume)
            print(("=> nasas warmup loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict, strict=False)
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        log_selection = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
        prec1 = _validate_once(val_loader, model, criterion, epoch, log_selection, tf_writer)

        # remember best prec@1 and save checkpoint
        if len(best_prec1s) == 0:
            best_prec1s.append(prec1)
            save_checkpoint_rank({
                'round': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
            }, n_rank=1, name='best')
        if len(worst_prec1s) == 0:
            worst_prec1s.append(prec1)
            save_checkpoint_rank({
                'round': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
            }, n_rank=1, name='worst')

        for r, prc in enumerate(best_prec1s):
            if prec1 > prc:
                for i in range(len(best_prec1s)-1, r-1, -1):
                    filename = '%s/%s/ckpt.best.%s.pth.tar' % (args.root_model, args.store_name, i+1)
                    shutil.copyfile(filename, filename.replace('{}.{}.pth.tar'.format('best', i+1),
                                                               '{}.{}.pth.tar'.format('best', i+2)))
                best_prec1s.insert(r, prec1)
                save_checkpoint_rank({
                    'round': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': prec1,
                }, n_rank=r + 1, name='best')
                if len(best_prec1s) > N_RANK:
                    best_prec1s.pop()
                break

        for r, prc in enumerate(worst_prec1s):
            if prec1 < prc:
                for i in range(len(worst_prec1s)-1, r-1, -1):
                    filename = '%s/%s/ckpt.worst.%s.pth.tar' % (args.root_model, args.store_name, i+1)
                    shutil.copyfile(filename, filename.replace('{}.{}.pth.tar'.format('worst', i+1),
                                                               '{}.{}.pth.tar'.format('worst', i+2)))
                worst_prec1s.insert(r, prec1)
                save_checkpoint_rank({
                    'round': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': prec1,
                }, n_rank=r + 1, name='worst')
                if len(worst_prec1s) > N_RANK:
                    worst_prec1s.pop()
                break
        # tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

        # output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        output_best = 'Best Prec@1: %.3f, Prec@1: %.3f\n' % (best_prec1s[0], prec1)
        print(output_best)
        log_selection.write(output_best + '\n')
        log_selection.flush()
        log_selection.close()


def finetune():
    global args, best_prec1, global_rank, local_rank
    args = parser.parse_args()
    arg_checker(args)

    if args.node_rank != 0:
        time.sleep(30)

    if args.philly_mpi_multi_node:
        global_rank = ompi_rank()
        local_rank = list(gpu_indices())[0]

        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=global_rank)
        print(
            "World Size is {}, Backend is {}, Init Method is {}, rank is {}".format(args.world_size, args.dist_backend,
                                                                                    args.dist_url, global_rank))
    elif args.philly_nccl_multi_node:
        torch.distributed.init_process_group(backend="nccl",
                                             init_method='env://')
        global_rank = torch.distributed.get_rank()
        local_rank = args.local_rank
    else:
        global_rank = -1
        local_rank = -1

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        torch.cuda.set_device(local_rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset + '_zip' if args.use_zip else args.dataset,
        args.modality,
        args.root_path)
    assert len([x for x in open(args.train_list)]) == args.training_size, "training size unmatched %d vs %d".format(
        len(args.train_list), args.training_size)

    full_arch_name = args.arch
    args.store_name = '_'.join(
        ['NAS_sptp', 'nasas' if args.enable_nasas else 'warmup', 'finetune',
         'ls{}'.format(args.weight_reg),
         args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs), 'droprate{}'.format(args.dropout),
         'num_dense_sample{}'.format(args.num_dense_sample),
         'dense_sample_stride{}'.format(args.dense_sample_stride)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.random_dense_sample_stride:
        args.store_name += '_randomstride'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        if global_rank == 0:
            check_rootfolders()
            time.sleep(5)
    else:
        check_rootfolders()

    if args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrain, num_classes=num_class, drop_rate=args.dropout)
    else:
        model = densenet121(num_classes=num_class, pretrained=args.pretrain, drop_rate=args.dropout)
    if (args.philly_mpi_multi_node or args.philly_nccl_multi_node) and args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    input_mean = model.mean
    input_std = model.std
    # policies = model.get_optim_policies()
    train_augmentation = get_train_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True,
        div=(args.arch not in ['BNInception', 'InceptionV3']),
        roll=(args.arch in ['BNInception', 'InceptionV3']))
    val_augmentation = get_val_augmentation(div=(args.arch not in ['BNInception', 'InceptionV3']),
                                            roll=(args.arch in ['BNInception', 'InceptionV3']))

    model = model.cuda()
    if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = torch.nn.parallel.DataParallel(model)

    optimizer = torch.optim.SGD(  # policies,
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=0.0 if args.enable_nasas else args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    dataset_train = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                               new_length=data_length,
                               modality=args.modality,
                               image_tmpl=prefix,
                               transform=torchvision.transforms.Compose([
                                   train_augmentation,
                                   # Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                   # ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                   normalize,
                               ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride,
                               num_dense_sample=args.num_dense_sample,
                               random_dense_sample_stride=args.random_dense_sample_stride, is_zip=args.use_zip)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train) if args.philly_mpi_multi_node or args.philly_nccl_multi_node else None
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)  # prevent something not % n_GPU

    dataset_val = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                             new_length=data_length,
                             modality=args.modality,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=torchvision.transforms.Compose([
                                 val_augmentation,
                                 # GroupScale(int(scale_size)),
                                 # GroupCenterCrop(crop_size),
                                 # Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                 # ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                 normalize,
                             ]), dense_sample=args.dense_sample, dense_sample_stride=args.dense_sample_stride,
                             num_dense_sample=args.num_dense_sample, is_zip=args.use_zip)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val) if args.philly_mpi_multi_node or args.philly_nccl_multi_node else None
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    '''
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    '''
    if args.evaluate:
        _validate_once(val_loader, model, criterion, 0)
        return

    with open(os.path.join(args.root_log, args.store_name, 'args' + str(global_rank) + '.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = None  # SummaryWriter(logdir=os.path.join(args.root_log, args.store_name))

    assert os.path.isfile(args.resume), "No checkpoint found in %s".format(args.resume)
    print(("=> nasas finetune loading checkpoint '{}'".format(args.resume)))
    checkpoint = torch.load(args.resume)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict, strict=True)
    print(("=> loaded checkpoint '{}' (epoch {})"
           .format(args.evaluate, -1)))

    if args.break_resume:
        args.resume = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
    log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
    _validate_once(val_loader, model, criterion, log_training)
    log_training.close()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
        # train for one epoch
        _train_once(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        log_training.close()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log_training = open(os.path.join(args.root_log, args.store_name, 'log' + str(global_rank) + '.csv'), 'a')
            prec1 = _validate_once(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            # output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            output_best = 'Best Prec@1: %.3f, Prec@1: %.3f\n' % (best_prec1, prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()
            if global_rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)
            log_training.close()


def _train_once(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    KL_losses = AverageMeter()
    crossentropy_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    if args.freeze_bn:
        def set_bn_eval(module):
            if 'BatchNorm' in type(module).__name__:
                module.eval()
                print('{} freezed'.format(type(module).__name__))
        model.apply(set_bn_eval)
    end = time.time()
    for i, (input, target) in enumerate(BackgroundGenerator(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)

        loss = criterion(output, target)
        crossentropy_losses.update(loss.item(), input.size(0))

        if args.enable_nasas and not args.finetune_mode:
            KL_CD = sum([getattr(m, 'KLreg', 0.0) for _, m in enumerate(model.modules())])
            KL_CD = KL_CD.cuda()
            loss += KL_CD
        else:
            KL_CD = None
        # measure accuracy and record loss

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        KL_losses.update(KL_CD.item() if KL_CD is not None else 0.0, input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Worker: {0}\t'
                      'Epoch: [{1}][{2}/{3}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'CrossEntropyLoss {celoss.val:.4f} ({celoss.avg:.4f})\t'
                      'KLLoss {klloss.val:.4f} ({klloss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(global_rank,
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, celoss=crossentropy_losses, klloss=KL_losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            if True:#global_rank == 0:
                log.write(output + '\n')
                log.flush()
    '''
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    '''


def _validate_once(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(BackgroundGenerator(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            if args.enable_nasas and not args.selection_mode:
                KL_CD = sum([getattr(m, 'KLreg', 0.0) for _, m in enumerate(model.modules())])
                KL_CD = KL_CD.cuda()
                loss += KL_CD

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if args.philly_mpi_multi_node or args.philly_nccl_multi_node:
                torch.distributed.all_reduce(tensor=loss, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(tensor=prec1, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(tensor=prec5, op=torch.distributed.ReduceOp.SUM)

                losses.update(loss.item()/float(torch.distributed.get_world_size()), input.size(0)*torch.distributed.get_world_size())
                top1.update(prec1.item()/float(torch.distributed.get_world_size()), input.size(0)*torch.distributed.get_world_size())
                top5.update(prec5.item()/float(torch.distributed.get_world_size()), input.size(0)*torch.distributed.get_world_size())
            else:
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if True:#global_rank == 0:
        if log is not None:
            log.write(output + '\n')
            log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def _test_once(test_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.permute([0, 2, 1, 3, 4]).contiguous()
            input = input.view([-1, args.num_dense_sample] + list(input.size()[-3:]))
            input = input.permute([0, 2, 1, 3, 4]).contiguous()

            input = input.cuda()
            target = target.cuda()
            # compute output
            output = []
            for s in range(input.size()[0]):
                this_output = model(input[s:s+1])
                output.append(torch.nn.functional.softmax(this_output, dim=-1))
            output = torch.mean(torch.cat(output, dim=0), dim=0, keepdim=True)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))

            top1.update(prec1.item(), 1)
            top5.update(prec5.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            outlog = ('Testing Results @ {}: Prec@1 {top1.val:.3f} Prec@5 {top5.val:.3f} Average {top1.avg}/{top5.avg}'
                      .format(i, top1=top1, top5=top5))
            print(outlog)

    outlog = ('Final Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Count {top1.count}'
              .format(top1=top1, top5=top5))
    print(outlog)
    if True:#global_rank == 0:
        if log is not None:
            log.write(outlog + '\n')
            log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def save_checkpoint_rank(state, n_rank, name):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)

    shutil.copyfile(filename, filename.replace('pth.tar', '{}.{}.pth.tar'.format(name, n_rank)))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    import torch.multiprocessing as tmp
    tmp.set_start_method('spawn')
    args = parser.parse_args()
    if args.selection_mode:
        selection()
        #finetune()
    elif args.test_mode:
        test()
    elif args.finetune_mode:
        finetune()
    else:
        train()