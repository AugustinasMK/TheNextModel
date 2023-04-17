import csv
import os.path as osp
import random
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


class GLV2Definition(object):
    def __init__(self, root, labels_file):
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
        self.labels_file = labels_file
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


class GLV2(Dataset):
    def __init__(self, df, subset='train', transform=None, augmentations=None, use_hnm=False):
        self.is_train = subset == 'train'
        self.is_gallery = subset == 'gallery'
        self.transform = transform
        self.augmentations = transform if augmentations is None else augmentations
        self.use_hnm = use_hnm

        if self.is_train:
            self.images = df.train
            label_to_images = {}
            image_to_labels = {}
            index = 0
            with open(df.labels_file, 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    index += 1
                    if index == 1:
                        continue
                    image_to_labels[row[0]] = row[1]
                    if row[1] in label_to_images:
                        label_to_images[row[1]].append(row[0])
                    else:
                        label_to_images[row[1]] = [row[0]]
            self.label_to_images = label_to_images
            self.image_to_labels = image_to_labels
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
            anchor_label = self.image_to_labels[name]

            positive_img = anchor_img

            semipositives = self.label_to_images[anchor_label]
            semipositives.remove(name)
            semipositive = random.choice(semipositives)
            semipositive_full_name = osp.join('/'.join(self.images[0][0].split('/')[:-1]), semipositive)
            semipositive_img = Image.open(semipositive_full_name)

            if self.use_hnm:
                if self.transform:
                    anchor_img = self.transform(anchor_img)
                    positive_img = self.augmentations(positive_img)
                    semipositive_img = self.augmentations(semipositive_img)
                return anchor_img, positive_img, semipositive_img, index, name
            else:
                negative_index = index
                while negative_index == index or self.image_to_labels[self.images[negative_index][1]] == anchor_label:
                    negative_index = random.randrange(len(self.images))
                negative_full_name, negative_name = self.images[negative_index]
                negative_img = Image.open(negative_full_name)

                if self.transform:
                    anchor_img = self.transform(anchor_img)
                    positive_img = self.augmentations(positive_img)
                    semipositive_img = self.augmentations(semipositive_img)
                    negative_img = self.augmentations(negative_img)
                return anchor_img, positive_img, semipositive_img, negative_img, name, anchor_label
        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img, name

    def get_negatives(self, positive_indexes: list, num_negatives: int = 2):
        positive_class_images = []
        for i in positive_indexes:
            full_name, name = self.images[i]
            i_class = self.image_to_labels[name]
            positive_class_images += self.label_to_images[i_class]

        pos_negative_indexes = []
        for i in range(len(self)):
            name = self.images[i][1]
            if name not in positive_class_images:
                pos_negative_indexes.append(i)

        negative_indexes = random.sample(pos_negative_indexes, num_negatives)
        negative_imgs = []
        for i in negative_indexes:
            full_name, name = self.images[i]
            negative_img = Image.open(full_name)
            if self.transform:
                negative_img = self.augmentations(negative_img)
            negative_imgs.append(negative_img)

        return torch.stack(negative_imgs)
