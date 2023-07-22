import os
import os.path as osp
import random


name2clsid = {
    'positive': '1',
    'negative': '0'
}

pos_ratio = 0.01

def gen_list(root):
    res = []
    cls_ids = os.listdir(root)
    pos_img_list = os.listdir(osp.join(root, 'positive'))
    neg_img_list = os.listdir(osp.join(root, 'negative'))
    random.shuffle(pos_img_list)
    pos_img_list = pos_img_list[:int(pos_ratio/(1-pos_ratio)*len(neg_img_list))]
    img_list = [osp.join(root, 'positive', i) + ' 1' for i in pos_img_list]
    img_list += [osp.join(root, 'negative', i) + ' 0' for i in neg_img_list]
    for i in img_list:
        if not i.split(' ')[0].endswith('.jpg'):
            continue
        res.append(i)
    return res

def main():
    for domain in ['drawings', 'neutral']:
        img_list = gen_list(domain)
        with open(domain + '_%.2f.txt'%pos_ratio, 'w') as f:
            for i in img_list:
                f.write(i + '\n')

if __name__ == '__main__':
    main()
