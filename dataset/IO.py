import torch.utils.data as data

from zipfile import ZipFile
from PIL import Image
import os
import numpy as np
from numpy.random import randint


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


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, num_dense_sample=32, dense_sample_stride=1, random_dense_sample_stride=False, is_zip=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.num_dense_sample = num_dense_sample
        self.dense_sample_stride = dense_sample_stride
        self.random_dense_sample_stride = random_dense_sample_stride
        self.is_zip = is_zip
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx, zip_f=None):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                if self.is_zip:
                    return [Image.open(zip_f.open(self.image_tmpl.format(idx))).convert('RGB')]
                else:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                if self.is_zip:
                    return [Image.open(zip_f.open(self.image_tmpl.format(1))).convert('RGB')]
                else:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [[' '.join(x[:-2]), x[-2], x[-1]] for x in tmp]
        if not self.test_mode or self.remove_missing:
            if self.test_mode and 'kinetics' in self.root_path:
                tmp = [item for item in tmp if int(item[1]) >= 32]
                print('####################### Heavy remove #######################')
            else:
                tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - self.num_dense_sample * self.dense_sample_stride)
            t_stride = self.num_dense_sample * self.dense_sample_stride // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - self.num_dense_sample * self.dense_sample_stride)
            t_stride = self.num_dense_sample * self.dense_sample_stride // self.num_segments
            #start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx = 0 if sample_pos == 1 else sample_pos//2
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - self.num_dense_sample * self.dense_sample_stride)
            t_stride = self.num_dense_sample * self.dense_sample_stride // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=2, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.is_zip:
            while not os.path.exists(full_path):
                print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
                if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                    file_name = self.image_tmpl.format('x', 1)
                    full_path = os.path.join(self.root_path, record.path, file_name)
                elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                    file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                    full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
                else:
                    file_name = self.image_tmpl.format(1)
                    full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        zip_f = ZipFile(os.path.join(self.root_path, record.path, 'RGB_frames.zip'), mode='r') if self.is_zip else None
        if self.dense_sample:
            assert self.num_segments == 1, "dense sample needs segment number to be 1."
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.num_dense_sample):
                    seg_imgs = self._load_image(record.path, p, zip_f)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.dense_sample_stride:
                        if self.random_dense_sample_stride and self.random_shift:
                            p += randint(1, self.dense_sample_stride+1)
                        else:
                            p += self.dense_sample_stride
        else:
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

        process_data = self.transform(images)
        if zip_f:
            zip_f.close()
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

if __name__ == '__main__':
    TSNDataSet('', '/data/home/v-yizzh/workspace/code/NAS_spatiotemporal/dataset/val_videofolder.txt')
