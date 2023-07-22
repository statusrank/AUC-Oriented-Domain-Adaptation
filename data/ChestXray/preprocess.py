import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def get_image_path(root, ext):
        res = []
        files = os.listdir(root)
        for file in files:
            if file.split('.')[-1] == ext:
                res.append(osp.join(root, file))
            elif osp.isdir(osp.join(root, file)):
                res += get_image_path(osp.join(root, file), ext)

        return res

def resize_chestxray_14(src_root='../ChestX-ray14', dst_root='./ChestX-ray14'):
    img_list = os.listdir(osp.join(src_root, 'images'))
    for i in tqdm(img_list):
        img = cv2.imread(osp.join(src_root, 'images', i))
        img = cv2.resize(img, (224,224))

        dir_path = osp.join(dst_root, 'images')
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(osp.join(dst_root, 'images', i), img)

def resize_chestxpert(src_root='../CheXpert', dst_root='./CheXpert'):
    img_list = get_image_path(src_root, 'jpg')
    for i in tqdm(img_list):
        dst = i.replace(src_root, dst_root)
        if osp.exists(dst):
            continue
        print(i)
        img = cv2.imread(i)
        img = cv2.resize(img, (224,224))
        dir_path = '/'.join(dst.split('/')[:-1])
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(dst, img)

name2label = {
    'Cardiomegaly': 0,
    'Edema': 1,
    'Consolidation': 2,
    'Atelectasis': 3,
    'Effusion': 4
}

def get_list_chestxray_14(src_root='../ChestX-ray14'):
    with open(osp.join(src_root, 'Data_Entry_2017.csv'), 'r') as f:
        lines = f.read().splitlines()
    
    anno = dict()
    for line in lines:
        line = line.split(',')
        name = line[0]
        label_str = line[1].split('|')
        label = ['0'] * 5
        for l in label_str:
            if l in name2label.keys():
                label[name2label[l]] = '1'
        anno[name] = ','.join(label)

    for split in ['test', 'train_val']:
        with open(osp.join(src_root, split + '_list.txt'), 'r') as f:
            lines = f.read().splitlines()

        split_list = []
        for line in lines:
            split_list.append(osp.join('ChestX-ray14/images', line) + ' ' + anno[line])
        with open('ChestX-ray14_' + split + '.txt', 'w') as f:
            f.write('\n'.join(split_list))

def get_list_chestxpert(src_root='../CheXpert'):
    for split in ['train', 'valid']:
        with open(osp.join(src_root, split + '.csv'), 'r') as f:
            lines = f.read().splitlines()
        title = lines[0].split(',')
        lines = lines[1:]

        idx2label = dict()
        for i in range(len(title)):
            if title[i] in name2label.keys():
                idx2label[i] = name2label[title[i]]

        res = []
        for line in lines:
            line = line.split(',')
            name = line[0]
            label = ['0'] * 5
            for i in range(len(line)):
                if i in idx2label.keys():
                    if len(line[i]) == 0:
                        label[idx2label[i]] = '0'
                    else:
                        label[idx2label[i]] = line[i].split('.')[0]
            res.append(name + ' ' + ','.join(label))
        res = '\n'.join(res)
        res = res.replace('CheXpert-v1.0-small', 'CheXpert')
        with open('CheXpert_' + split + '.txt', 'w') as f:
            f.write(res)

def load_list(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        name, label = line.split(' ')
        label = [int(i) for i in label.split(',')]
        data.append([name, label])
    return data

def resample():
    data_14 = load_list('ChestX-ray14_train.txt')
    data_xpert = load_list('CheXpert_train_ori.txt')

    data_xpert_clean = []
    for i in range(len(data_xpert)):
        name, label = data_xpert[i]
        if -1 in label:
            continue
        data_xpert_clean.append(data_xpert[i])
    data_xpert = data_xpert_clean

    def cnt_cls(data, c):
        cnt_pos = 0
        for i in range(len(data)):
            cnt_pos += data[i][1][c]
        return cnt_pos, len(data) - cnt_pos, cnt_pos / len(data)

    num_classes = len(data_14[0][1])
    mark_del = [False] * len(data_xpert)
    for i in range(num_classes):
        _,_,pos_ratio_14 = cnt_cls(data_14, i)
        _,_,pos_ratio_xpert = cnt_cls(data_xpert, i)
        print(pos_ratio_14, pos_ratio_xpert)
        for j in range(len(data_xpert)):
            if data_xpert[j][1][i] == 1 and np.random.rand() > pos_ratio_14 / pos_ratio_xpert:
                mark_del[j] = True

    data_xpert_clean = []
    for i in range(len(data_xpert)):
        if mark_del[i]:
            continue
        data_xpert_clean.append(data_xpert[i])
    data_xpert = data_xpert_clean

    num_classes = len(data_14[0][1])
    for i in range(num_classes):
        print('='*30)
        print(cnt_cls(data_14, i))
        print(cnt_cls(data_xpert, i))

    # with open('CheXpert_train.txt', 'w') as f:
    #     for line in data_xpert:
    #         f.write(line[0]+' '+','.join([str(i) for i in line[1]])+'\n')

def main():
    # resize_chestxray_14()
    # resize_chestxpert()
    # get_list_chestxray_14()
    # get_list_chestxpert()
    resample()

if __name__ == '__main__':
    main()
