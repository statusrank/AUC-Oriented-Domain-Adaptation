import cv2
import os
import os.path as osp
import random
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

def resize(src_root, dst_root):
    img_list = get_image_path(src_root, 'jpg')
    for i in tqdm(img_list):
        dst = i.replace(src_root, dst_root)
        if osp.exists(dst):
            continue
        img = cv2.imread(i)
        img = cv2.resize(img, (224,224))
        dir_path = '/'.join(dst.split('/')[:-1])
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(dst, img)

def get_list_SIIM(src_root='./SIIM-ISIC', dst_root='./SIIM-ISIC'):
    name2label = {
        'benign': '0',
        'malignant': '1'
    }

    with open(osp.join(src_root, 'train.csv'), 'r') as f:
        lines = f.read().splitlines()[1:]

    img_list_cls = {}
    for c in name2label.keys():
        img_list_cls[c] = []
    anno = dict()
    for line in lines:
        line = line.split(',')
        name = line[0]
        label = name2label[line[6]]
        anno[name] = label
        img_list_cls[line[6]].append(osp.join(dst_root, 'images/train', name+'.jpg') + ' ' + label)

    img_list_train = []
    img_list_val = []
    for c in name2label.keys():
        img_list = img_list_cls[c]
        random.shuffle(img_list)
        # if c == 'benign':
        #     img_list = img_list[:int(10015/1627*len(img_list_cls['malignant']))]
        n_train = int(0.8 * len(img_list))
        img_list_train += img_list[:n_train]
        img_list_val += img_list[n_train:]
    
    for img_list, split in zip(
        [img_list_train, img_list_val, img_list_train + img_list_val], 
        ['train', 'val', 'trainval']):
        with open('SIIM_' + split + '.txt', 'w') as f:
            f.write('\n'.join(img_list))

def get_list_HAM10000(src_root='../HAM10000', dst_root='./HAM10000'):
    name2label = {
        'bkl': '0',
        'nv': '0',
        'df': '0',
        'vasc': '0',
        'akiec': '0',
        'bcc': '1',
        'mel': '1'
    }

    with open(osp.join(src_root, 'HAM10000_metadata'), 'r') as f:
        lines = f.read().splitlines()[1:]

    img_list_cls = {}
    for c in name2label.keys():
        img_list_cls[c] = []
    # anno = dict()
    for line in lines:
        line = line.split(',')
        name = line[1]
        label = name2label[line[2]]
        # anno[name] = label
        img_list_cls[line[2]].append(osp.join(dst_root, 'images', name+'.jpg') + ' ' + label)

    img_list_train = []
    img_list_val = []
    for c in name2label.keys():
        img_list = img_list_cls[c]
        random.shuffle(img_list)
        n_train = int(0.8 * len(img_list))
        img_list_train += img_list[:n_train]
        img_list_val += img_list[n_train:]
    
    for img_list, split in zip(
        [img_list_train, img_list_val, img_list_train + img_list_val], 
        ['train', 'val', 'trainval']):
        with open('HAM10000_' + split + '.txt', 'w') as f:
            f.write('\n'.join(img_list))

def load_list(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        name, label = line.split(' ')
        label = [int(i) for i in label.split(',')]
        data.append([name, label])
    return data

# def resample():
#     data_14 = load_list('ChestX-ray14_train.txt')
#     data_xpert = load_list('CheXpert_train.txt')
    
#     def cnt_cls(data, c):
#         cnt_pos = 0
#         for i in range(len(data)):
#             cnt_pos += data[i][1][c]
#         return cnt_pos, len(data) - cnt_pos, cnt_pos / len(data)

#     num_classes = len(data_14[0][1])
#     for i in range(num_classes):
#         print('='*30)
#         print(cnt_cls(data_14, i))
#         print(cnt_cls(data_xpert, i))

def main():
    # resize('../HAM10000', './HAM10000')
    # resize('../SIIM-ISIC', './SIIM-ISIC')
    get_list_HAM10000()
    get_list_SIIM()
    # resample()

if __name__ == '__main__':
    main()
