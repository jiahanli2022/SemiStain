import os.path
import numpy as np
import torch
import json

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import random
import util.util as util
import torchvision

class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.opt = opt

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'


        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B


        if opt.phase == 'train':
            # pair consistency check
            if self.opt.use_semistain:
                self.A_psnr_dict = self.read_psnr_txt(opt.psnr_txt)
                A_psnrs = sorted(self.A_psnr_dict.values(), reverse=True)
                self.A_psnr_threshold = A_psnrs[int(len(A_psnrs) * opt.psnr_threshold / 100.) - 1]
                print('A_psnr_threshold', self.A_psnr_threshold)

    def read_psnr_txt(self, psnr_txt):
        A_psnr_dict = {}

        with open(psnr_txt, 'r') as f:
            for line in f.readlines():
                content = line.rstrip()
                A_path, psnr = content.split('=')[1:]
                A_path = A_path.strip().split(',')[0]
                psnr = float(psnr)
                A_psnr_dict[A_path] = psnr
        return A_psnr_dict

    def __getitem__(self, index):
        if self.opt.serial_batches:
            index_B = index % self.A_size
        else:
            index = random.randint(0, self.A_size - 1)
            index_B = index % self.B_size
            
        A_path = self.A_paths[index]
        B_path = self.B_paths[index_B]



        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        start_x = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
        start_y = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
        A_img = A_img.crop((start_x, start_y, start_x + self.opt.crop_size, start_y + self.opt.crop_size))
        B_img = B_img.crop((start_x, start_y, start_x + self.opt.crop_size, start_y + self.opt.crop_size))

 
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        A = transform(A_img)
        B = transform(B_img)

        if not self.opt.no_flip and self.phase == 'train':
            if random.random() > 0.5:
                A = torchvision.transforms.functional.hflip(A)
                B = torchvision.transforms.functional.hflip(B)


        isConsistent = (self.opt.use_semistain and self.phase == 'train' and self.A_psnr_dict[A_path] > self.A_psnr_threshold)


        if self.opt.use_semistain:
            A_aug = A
        else:
            A_aug = []

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'isConsistent': isConsistent, 'A_aug': A_aug}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size