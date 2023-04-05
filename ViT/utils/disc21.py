from torch.utils.data import Dataset
import os.path as osp
from glob import glob
from PIL import Image
import random
import torch


class DISC21Definition(object):
    def __init__(self, root):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'validation')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_pids = 0
        self.num_gallery_pids = 0
        self.num_query_pids = 0
        self.has_time_info = False
        self.load()

    def preprocess(self, splitter='T', file_paths=None):
        if file_paths is None:
            file_paths = self.train_dir
            fpaths = glob(osp.join(self.train_dir, '*.jpg'))
        else:
            fpaths = glob(osp.join(file_paths, '*.jpg'))
        data = []
        all_pids = {}
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid = int(fname[:-4].split(splitter)[1])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            data.append((file_paths + '/' + fname, fname))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_pids = self.preprocess('T', self.train_dir)
        self.gallery, self.num_gallery_pids = self.preprocess('R', self.gallery_dir)
        self.query, self.num_query_pids = self.preprocess('Q', self.query_dir)
        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:6d} | {:8d}".format(self.num_train_pids, len(self.train)))
        print("  gallery  | {:6d} | {:8d}".format(self.num_gallery_pids, len(self.gallery)))
        print("  query    | {:6d} | {:8d}".format(self.num_query_pids, len(self.query)))


class DISC21(Dataset):
    def __init__(self, df, subset='train', transform=None, augmentations=None):
        self.is_train = subset == 'train'
        self.is_gallery = subset == 'gallery'
        self.transform = transform
        self.augmentations = transform if augmentations is None else augmentations

        if self.is_train:
            self.images = df.train
        elif self.is_gallery:
            self.images = df.gallery
        else:
            self.images = df.query

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        full_name, name = self.images[index]
        anchor_img = Image.open(full_name)

        if self.is_train:
            positive_img = anchor_img
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.augmentations(positive_img)
            return anchor_img, positive_img, index, name
        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img, name

    def get_negatives(self, positive_indexes: list, num_negatives: int = 2):
        pos_negative_indexes = []
        for i in range(len(self)):
            if i not in positive_indexes:
                pos_negative_indexes.append(i)

        for i in pos_negative_indexes:
            if i in positive_indexes:
                raise Exception('Negative index is in positive indexes')

        negative_indexes = random.sample(pos_negative_indexes, num_negatives)
        negative_imgs = []
        for i in negative_indexes:
            full_name, name = self.images[i]
            negative_img = Image.open(full_name)
            if self.transform:
                negative_img = self.augmentations(negative_img)
            negative_imgs.append(negative_img)

        return torch.stack(negative_imgs)
