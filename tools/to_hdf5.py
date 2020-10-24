import sys
import os

import h5py
from zipfile import ZipFile
import numpy as np

from PIL import Image


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Converter(object):
    def __init__(self, root_path, list_file, target_path, image_tmpl):
        self._list_file = list_file
        self._root_path = root_path
        self._target_path = target_path
        self._image_tmpl = image_tmpl

        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self._list_file)]
        tmp = [[' '.join(x[:-2]), x[-2], x[-1]] for x in tmp]
        self._video_list = [VideoRecord(item) for item in tmp]

        if self._image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self._video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self._video_list)))

    def _full_path(self, directory, idx):
        return os.path.join(self._root_path, directory, self._image_tmpl.format(idx))

    def _load_image(self, directory, idx):
        try:
            return Image.open(os.path.join(self._root_path, directory, self._image_tmpl.format(idx))).convert('RGB')
        except Exception:
            print('error loading image:', os.path.join(self._root_path, directory, self._image_tmpl.format(idx)))
            return Image.open(os.path.join(self._root_path, directory, self._image_tmpl.format(1))).convert('RGB')

    def convert(self):
        raise NotImplementedError()


class HDF5Converter(Converter):
    def __init__(self, root_path, list_file, target_path, image_tmpl):
        super(HDF5Converter, self).__init__(root_path, list_file, target_path, image_tmpl)

    def convert(self):
        for record in self._video_list:
            if not os.path.exists(os.path.join(self._target_path, record.path)):
                os.makedirs(os.path.join(self._target_path, record.path))
            assert not os.path.exists(os.path.join(self._target_path, record.path, 'RGB_frames')), "{} already exist".format(os.path.join(self._target_path, record.path, 'RGB_frames'))
            with h5py.File(os.path.join(self._target_path, record.path, 'RGB_frames'), 'w') as hdf:
                for idx in range(record.num_frames):
                    img = np.asarray((self._load_image(record.path, idx+1)), dtype="uint8")
                    hdf.create_dataset(record.path+"/"+self._image_tmpl.format(idx+1), data=img, dtype="uint8")
            print("{} Done".format(record.path))


class ZIPConverter(Converter):
    def __init__(self, root_path, list_file, target_path, image_tmpl):
        super(ZIPConverter, self).__init__(root_path, list_file, target_path, image_tmpl)

    def convert(self):
        _video_num = len(self._video_list)
        for i, record in enumerate(self._video_list):
            if not os.path.exists(os.path.join(self._target_path, record.path)):
                os.makedirs(os.path.join(self._target_path, record.path))
            assert not os.path.exists(os.path.join(self._target_path, record.path, 'RGB_frames.zip')), "{} already exist".format(os.path.join(self._target_path, record.path, 'RGB_frames.zip'))
            with ZipFile(os.path.join(self._target_path, record.path, 'RGB_frames.zip'), 'w') as zipf:
                for idx in range(record.num_frames):
                    #img = np.asarray((self._load_image(record.path, idx+1)), dtype="uint8")
                    zipf.write(self._full_path(record.path, idx+1), arcname=self._image_tmpl.format(idx+1))
            print("{} of {} ({}) Done".format(str(i), str(_video_num), record.path))


def main(list_file):
    root_path = "/home/sda/data-writable/something-something/"
    target_path = "/home/sdb/writable/20bn-something-something-v1_zip/"
    #list_file = "/home/sda/data-writable/kinetics400_frame/val_videofolder.txt"
    list_file = os.path.join(root_path, list_file)

    cvt = ZIPConverter(os.path.join(root_path, '20bn-something-something-v1'), list_file, target_path, '{:05d}.jpg')
    cvt.convert()


if __name__ == '__main__':
    main(sys.argv[1])

