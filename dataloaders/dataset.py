from .base_dataset import BaseDataset
from .base_dataset import pil_loader
import numpy as np
import io
import os
import os.path as osp
import torch
# from utils.logger import logger
import random
import cv2
import lmdb
from tqdm import tqdm
import pickle
import fcntl
from PIL import Image
from torch.utils.data import Sampler
from collections import Counter
import pandas as pd
import math
from sklearn.utils import shuffle

def build_lmdb(save_path, metas, commit_interval=1000):
    with open('lock', 'w') as f:
        if not save_path.endswith('.lmdb'):
            raise ValueError("lmdb_save_path must end with 'lmdb'.")

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        if osp.exists(save_path):
            # print('Folder [{:s}] already exists.'.format(save_path))
            return

        if not osp.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))

        data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
        data_size = data_size_per_img * len(metas)
        env = lmdb.open(save_path, map_size=data_size * 10)
        txn = env.begin(write=True)

        shape = dict()

        print('Building lmdb...')
        for i in tqdm(range(len(metas))):
            image_filename = metas[i][0]
            img = pil_loader(filename=image_filename)
            assert img is not None and len(img.shape) == 3

            txn.put(image_filename.encode('ascii'), img.copy(order='C'))
            shape[image_filename] = '{:d}_{:d}_{:d}'.format(img.shape[0], img.shape[1], img.shape[2])

            if i % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        pickle.dump(shape, open(os.path.join(save_path, 'meta_info.pkl'), "wb"))

        txn.commit()
        env.close()
        print('Finish writing lmdb.')

class Dataset(BaseDataset):
    def __init__(self, args, list_file, split='train', **kwargs):
        super().__init__(args)
        self.data_dir = osp.join(args.root, args.data)
        self.split = split
        _data_list = [os.path.join(self.data_dir, i + '.txt') for i in list_file]
        # _data_list = os.path.join(self.data_dir, list_file + '.txt')
        self._data_list = _data_list
        if split == 'train':
            self.transform = self.transform_train()
        elif split == 'val' or split == 'test':
            self.transform = self.transform_validation()
        else:
            raise ValueError

        lines = []
        for l in _data_list:
            with open(l) as f:
                lines += f.read().splitlines()

        self.metas = []
        _labels = []
        _labels_pn = []
        _freq_info = None
        for line in lines:
            path, label = line.split(' ')
            if args.multi_label:
                label = np.array([int(i) for i in label.split(',')])
                _labels_pn.append(max(label))
                self.num_classes = len(label)
                
                if _freq_info is None:
                    _freq_info = np.asarray([0.] * self.num_classes)
                
                _freq_info += label
            else:
                label = int(label)
                _labels_pn.append(int(label))
                self.num_classes = 2
            path = osp.join(self.data_dir, path)
            _labels.append(label)
            self.metas.append((path, label))

        self.labels = _labels

        if args.multi_label:
            self._freq_info = []
            for i in range(self.num_classes):
                # self._freq_info.append(
                #     sum([meta[1][i] / len(self.metas) for meta in self.metas])
                # )
                self._freq_info.append(_freq_info[i] * 1.0 / sum(_freq_info))
                
        else:
            _cls_num_list = pd.Series(_labels_pn).value_counts().sort_index().values
            self._freq_info = [
                num * 1.0 / sum(_cls_num_list) for num in _cls_num_list
            ]

        self.num = len(self.metas)

        # logger.info('%s set (%s) has %d samples per epoch' % (self.split, list_file, self.__len__()))

        if self.args.lmdb_dir is not None:
            self._load_image = self._load_image_lmdb
        else:
            self._load_image = self._load_image_pil

        self.initialized = False

    def _init_lmdb(self):
        if not self.initialized:
            lmdb_dir = osp.join(self.args.lmdb_dir, '+'.join([i.split('/')[-1].split('.')[0] for i in self._data_list]) + '.lmdb')
            build_lmdb(lmdb_dir, self.metas)
            env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(open(os.path.join(lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def __len__(self):
        return self.num

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)

    def _load_image_pil(self, filename):
        img = pil_loader(filename=filename)
        return img

    def _load_image_lmdb(self, filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        return img

    def __getitem__(self, idx):
        image = Image.fromarray(self._load_image(self.metas[idx][0]))
        label = self.metas[idx][1]
        image = self.transform(image)
        return image, label

    def get_labels(self):
        return self.labels
    
    def get_freq_info(self):
        return self._freq_info

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        self.class_vector = class_vector
        # self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size

        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)
        y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})
        self.class_batch_size = {
            k: math.ceil(n * batch_size / y.shape[0])
            for k, n in y_counter.items()
        }
        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n)

        # sampling for each batch
        data = self.data.copy()
        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(
                    batch) == self.real_batch_size, 'not enough instances!'
            except (ValueError, AssertionError):
                break
            # print('sampled a batch ...')
            result.extend(shuffle(batch.index))
            data.drop(index=batch.index, inplace=True)
        return result

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class StratifiedMultiLabelSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        self.class_vector = class_vector
        # self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size

        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)

        self.idx_per_cls = [np.where(y.sum(-1) == 0)[0]]
        self.class_batch_size = [0]
        self.real_batch_size = 0
        self.num_classes = y.shape[1]

        for i in range(self.num_classes):
            pos_idx = np.where(y[:,i] == 1)[0]
            neg_idx = np.where(y[:,i] == 1)[0]
            self.idx_per_cls.append(pos_idx)
            # self.class_batch_size.append(
            #     {
            #         'pos': math.ceil(len(pos_idx) / len(y) * batch_size),
            #         'neg': math.ceil(len(neg_idx) / len(y) * batch_size)
            #     }
            # )
            self.class_batch_size.append(
                math.ceil(len(pos_idx) / len(y) * batch_size)
            )
            self.real_batch_size += self.class_batch_size[-1]

        self.class_batch_size[0] = max(0, batch_size - self.real_batch_size)
        self.real_batch_size = max(batch_size, self.real_batch_size)

        # print(self.class_batch_size)

        # y_counter = Counter(y)
        # self.data = pd.DataFrame({'y': y})
        # self.class_batch_size = {
        #     k: math.ceil(n * batch_size / y.shape[0])
        #     for k, n in y_counter.items()
        # }
        # self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n)

        # sampling for each batch
        idx_per_cls = self.idx_per_cls.copy()
        result = []

        while True:
            batch = []
            for i in range(0, 1 + self.num_classes):
                idx = idx_per_cls[i]
                if len(idx) < self.class_batch_size[i]:
                    return result
                select = np.random.choice(range(len(idx)), self.class_batch_size[i], replace=False)
                batch += list(idx[select])
                idx_per_cls[i] = np.delete(idx, select)
            
            shuffle(batch)
            result += batch
        
    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)