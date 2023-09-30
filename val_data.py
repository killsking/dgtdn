"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import random
import numpy as np

from random import randrange
from torchvision.transforms import functional as FF

class TrainDataNH(data.Dataset):
    def __init__(self, val_data_dir='/data/zsd/NH-HAZE/train_NH/'):
        super().__init__()
        val_list = val_data_dir + 'trainlist.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        crop_width, crop_height = [240,240]
        index=index%len(self.haze_names)
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'haze/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear_images/' + gt_name)
        width, height = haze_img.size
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        rand_rot=random.randint(0,3)
        if rand_rot:
           haze_crop_img=FF.rotate(haze_crop_img,90*rand_rot)
           gt_crop_img=FF.rotate(gt_crop_img,90*rand_rot)
        rand_rot=random.randint(0,20)   
        if rand_rot<2:
           tran=random.randint(0,20)/60+0.6
           
           
           haze_crop_img = np.array(haze_crop_img)
           for i in range(3):
               A=random.randint(0,75)+180
               haze_crop_img[:,:,i]=haze_crop_img[:,:,i]*tran+A*(1-tran)
           #print(haze_crop_img.shape)
           haze_crop_img=Image.fromarray(np.uint8(haze_crop_img))
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Transform to tensor --- #
        

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)*500
        


class ValDataNH(data.Dataset):
    def __init__(self, val_data_dir='/data/zsd/NH-HAZE/valid_NH/'):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'haze/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear_images/' + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
        


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)
        width, height = haze_img.size
        
        new_width,new_height=int(width/16)*16,int(height/16)*16
        
        haze_img=haze_img.resize((new_width, new_height),Image.ANTIALIAS)
        gt_img=gt_img.resize((new_width, new_height),Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

class ValDataR(data.Dataset):
    def __init__(self, val_data_dir, baseSize=1):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.baseSize=baseSize

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)
        if self.baseSize>1:
           width, height = haze_img.size
           width,height=int(width/self.baseSize+1)*self.baseSize, int(height/self.baseSize+1)*self.baseSize
           haze_img=haze_img.resize((width, height),Image.ANTIALIAS)
           gt_img =gt_img.resize((width, height),Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
        

class ValData512(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)
        
        
        
        haze_img = haze_img.resize((512, 512),Image.ANTIALIAS)
        gt_img = gt_img.resize((512, 512),Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_all = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        haze = transform_all(haze_img)
        gt = transform_all(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
        
        
        
                


class ValDataN(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)