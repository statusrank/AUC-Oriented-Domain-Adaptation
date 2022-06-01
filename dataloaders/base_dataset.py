import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from abc import ABC, abstractmethod
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename, label=False):
    ext = os.path.splitext(filename)[-1]
    ext = ext.lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            # img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
        return img
    elif ext == '.npy':
        data = np.load(filename, allow_pickle=True)
        return data.T.reshape((1, 64, -1))[:,:,:1000]
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass
    
    @staticmethod
    def modify_commandline_options(parser,istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train(self):
        if self.args.is_cen:
            transformer = transforms.Compose([
                transforms.Resize(self.args.resize_size),
                transforms.Scale(self.args.resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        else:
            transformer = transforms.Compose([
                  transforms.Resize(self.args.resize_size),
                  transforms.RandomResizedCrop(self.args.crop_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        return transformer

    def transform_validation(self):
        transformer = transforms.Compose([
            transforms.Resize(self.args.resize_size),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        return transformer
