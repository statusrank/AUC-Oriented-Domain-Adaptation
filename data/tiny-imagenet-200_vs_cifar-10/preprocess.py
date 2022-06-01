import os
import os.path as osp
import pickle
import numpy as np
import cv2
from tqdm import tqdm


def transform_cifar():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_train_data():
        labels = []
        filenames = []
        data = []

        for i in range(1, 6):
            mp = unpickle('cifar-10/cifar-10-batches-py/data_batch_%d'%i)
            labels += mp[b'labels']
            data.append(mp[b'data'])
            filenames += mp[b'filenames']

        data = np.concatenate(data, 0)
        return data, labels, filenames

    def load_val_data():
        mp = unpickle('cifar-10/cifar-10-batches-py/test_batch')
        return mp[b'data'], mp[b'labels'], mp[b'filenames']

    data, labels, filenames = load_train_data()
    # data, labels, filenames = load_val_data()
    lines = []
    for i in tqdm(range(len(data))):
        dst = osp.join('images', str(filenames[i])[2:-1])
        img = data[i].reshape((3, 32,32))
        img = np.transpose(img, (1,2,0))
        img = img[:,:,::-1]
        cv2.imwrite(dst, img)
        lines.append('cifar-10/images/' + str(filenames[i])[2:-1] + ' ' + str(labels[i]))

    with open('cifar-10/cifar-10_train_ori.txt', 'w') as f:
        f.write('\n'.join(lines))

def gen_list_binary_cifar(split, pos_cls='2'):
    with open('cifar-10/%s_list_ori.txt'%split, 'r') as f:
        lines = f.read().splitlines()

    lines_binary = []
    for line in lines:
        line = line.split(' ')
        if line[1] == pos_cls:
            line[1] = '1'
        else:
            line[1] = '0'
        lines_binary.append(' '.join(line))

    with open('cifar-10_%s.txt'%split, 'w') as f:
        f.write('\n'.join(lines_binary))


def name_translate():
    with open('tiny-image-200/imagenet_cls_names.txt', 'r') as f:
        lines = f.read().splitlines()
    cls2name_cn = dict()
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split(' ')
        clss, name_cn, name = line[1], line[2].replace(',', ''), ' '.join(line[3:])
        cls2name_cn[clss] = name_cn
    return cls2name_cn

def gen_name_list():
    cls2name_cn = name_translate()
    with open('tiny-image-200/words.txt', 'r') as f:
        lines = f.read().splitlines()
    cls2name = dict()
    for line in lines:
        line = line.split('\t')
        cls2name[line[0]] = line[1]

    with open('tiny-image-200/wnids.txt', 'r') as f:
        lines = f.read().splitlines()

    with open('tiny-image-200/cls_names.txt', 'w') as f:
        for line in lines:
            f.write(line + '\t' + cls2name[line] + '\t' + cls2name_cn[line] + '\n')

def gen_list_train():
    clss_list = os.listdir('train')
    lines = []
    for clss in clss_list:
        img_list = os.listdir(osp.join('tiny-image-200/train', clss, 'images'))
        for i in img_list:
            lines.append(osp.join('tiny-image-200/train', clss, 'images', i) + ' ' + clss)
    with open('tiny-image-200/train_list_ori.txt', 'w') as f:
        f.write('\n'.join(lines))

def gen_list_val():
    with open('tiny-image-200/val/val_annotations.txt', 'r') as f:
        lines = f.read().splitlines()
    lines = ['val/images/' + ' '.join(line.split('\t')[:2]) for line in lines]

    with open('tiny-image-200/val_list_ori.txt', 'w') as f:
        f.write('\n'.join(lines))

def gen_list_binary_imagenet(split, pos_cls='鸟'):
    cls2name_cn = name_translate()
    with open('tiny-image-200/%s_list_ori.txt'%split, 'r') as f:
        lines = f.read().splitlines()

    lines_binary = []
    for line in lines:
        line = line.split(' ')
        if cls2name_cn[line[1]] == pos_cls:
            line[1] = '1'
        else:
            line[1] = '0'
        lines_binary.append(' '.join(line))

    with open('tiny-imagenet-200_%s'%split, 'w') as f:
        f.write('\n'.join(lines_binary))


if __name__ == '__main__':
    # gen_list_train()
    # gen_list_val()
    for split in ['train', 'val']:
        gen_list_binary_imagenet(split)
        gen_list_binary_cifar(split)