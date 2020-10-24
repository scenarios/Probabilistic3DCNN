import os

ROOT_DATASET = '/mnt/data/'


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_zip(modality, root_path=''):
    assert modality == 'RGB', "Currently RGB only."
    filename_categories = 'ucf101_zip/classInd.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_path, 'ucf101_zip/')
        filename_imglist_train = 'ucf101_zip/train_videofolder.txt'
        filename_imglist_val = 'ucf101_zip/val_videofolder.txt'
        prefix = 'image_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality, root_path=''):
    assert modality == 'RGB', "Currently RGB only."
    filename_categories = '20bn-something-something-v1/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_path, '20bn-something-something-v1/')
        filename_imglist_train = '20bn-something-something-v1/train_videofolder.txt'
        filename_imglist_val = '20bn-something-something-v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = os.path.join(root_path, 'something/v1/20bn-something-something-v1-flow/')
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something_zip(modality, root_path=''):
    assert modality == 'RGB', "Currently RGB only."
    filename_categories = '20bn-something-something-v1_zip/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_path, '20bn-something-something-v1_zip/')
        filename_imglist_train = '20bn-something-something-v1_zip/train_videofolder.txt'
        filename_imglist_val = '20bn-something-something-v1_zip/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = os.path.join(root_path, 'something/v1/20bn-something-something-v1-flow/')
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality, root_path=''):
    assert modality == 'RGB', "Currently RGB only."
    filename_categories = '20bn-something-something-v2/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_path, '20bn-something-something-v2/')
        filename_imglist_train = '20bn-something-something-v2/train_videofolder.txt'
        filename_imglist_val = '20bn-something-something-v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2_zip(modality, root_path=''):
    assert modality == 'RGB', "Currently RGB only."
    filename_categories = '20bn-something-something-v2_zip/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_path, '20bn-something-something-v2_zip/')
        filename_imglist_train = '20bn-something-something-v2_zip/train_videofolder.txt'
        filename_imglist_val = '20bn-something-something-v2_zip/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality, root_path=''):
    filename_categories = 400
    if modality == 'RGB':
        root_data = os.path.join(root_path, 'kinetics400_frame/')
        filename_imglist_train = 'kinetics400_frame/train_videofolder.txt'
        filename_imglist_val = 'kinetics400_frame/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics_zip(modality, root_path=''):
    filename_categories = 400
    if modality == 'RGB':
        root_data = os.path.join(root_path, 'kinetics400_frame_zip/')
        filename_imglist_train = 'kinetics400_frame_zip/train_videofolder.txt'
        filename_imglist_val = 'kinetics400_frame_zip/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality, root_path=''):
    dict_single = {'jester': return_jester, 'something': return_something, 'something_zip': return_something_zip,
                   'somethingv2': return_somethingv2, 'somethingv2_zip': return_somethingv2_zip,
                   'ucf101': return_ucf101, 'ucf101_zip': return_ucf101_zip, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'kinetics_zip': return_kinetics_zip}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality, root_path)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(root_path, file_imglist_train)
    file_imglist_val = os.path.join(root_path, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(root_path, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix