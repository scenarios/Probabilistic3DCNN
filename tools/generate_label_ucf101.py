# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1

import os

root_path = '/home/sda/data-writable/ucf101/'

if __name__ == '__main__':
    os.chdir(root_path)
    dataset_name = 'ucf101'
    with open('%s-labels.txt' % dataset_name) as f:
        lines = f.readlines()
    dict_categories = {}
    for line in lines:
        line = line.rstrip()
        line = line.split( )
        dict_categories[line[1]] = int(line[0])-1


    files_input = ['testlist01.txt', 'trainlist01.txt']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split('.')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[0].split('/')[0]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('./UCF-101_image', curFolder, 'i'))
            output.append('%s %d %d' % (os.path.join('', curFolder), len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))