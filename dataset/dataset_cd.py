import os, sys
sys.path.append(os.getcwd())

from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw, ImageOps
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid 
from io import BytesIO
import random
import math

# from .tsv import TSVFile

from io import BytesIO
import base64
from PIL import Image
import numpy as np

from ldm.util import instantiate_from_config


class LEVIRDataset(BaseDataset):
    def __init__(self, 
                root_dir,  # /disk3/zeyu/LEVIR-CD/
                image_size=512,
                random_crop = True,
                random_flip = True,
                train_val_test = 'train',  # wzy
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.root_dir = root_dir
        # self.transform = transform
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()
        self.train_val_test = train_val_test

        self.images_A = os.listdir(os.path.join(root_dir, train_val_test, 'A'))
        self.images_B = os.listdir(os.path.join(root_dir, train_val_test, 'B'))
        self.labels = os.listdir(os.path.join(root_dir, train_val_test, 'label'))

        # self.random_rotate = True

        # self.transform = transforms.Compose([
        #     # transforms.RandomRotation(degrees=10),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Resize(256),
        #     transforms.ToTensor()
        # ])


    def total_images(self):
        return min(len(self.images_A), len(self.images_B), len(self.labels))


    def __len__(self):
        return min(len(self.images_A), len(self.images_B), len(self.labels))


    def __getitem__(self, index):
        out = {}

        out['id'] = index

        img_name = self.images_A[index]

        out['item'] = img_name

        img_A = Image.open(os.path.join(self.root_dir, self.train_val_test, 'A', img_name)).convert('RGB')
        img_B = Image.open(os.path.join(self.root_dir, self.train_val_test, 'B', img_name)).convert('RGB')
        label = Image.open(os.path.join(self.root_dir, self.train_val_test, 'label', img_name)).convert("L")

        if self.random_crop:
            width, height = img_A.size

            # 计算最小边需要调整到达512以上的缩放比例
            scale_factor = max(self.image_size / width, self.image_size / height) * random.uniform(1 , 1 / 0.8)
            
            # 调整图片大小
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_A = img_A.resize((new_width, new_height), Image.NEAREST)
            img_B = img_B.resize((new_width, new_height), Image.NEAREST)
            label = label.resize((new_width, new_height), Image.NEAREST)

            # 随机裁剪出512*512大小的图片
            crop_x = random.randint(0, new_width - self.image_size)
            crop_y = random.randint(0, new_height - self.image_size)

            img_A = img_A.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            img_B = img_B.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            label = label.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))

        else:
            img_A = img_A.resize((512, 512), Image.NEAREST)
            img_B = img_B.resize((512, 512), Image.NEAREST)
            label = label.resize((512, 512), Image.NEAREST)

        # if self.random_flip and random.random()<0.5:
        #     # flip = True
        #     img_A = ImageOps.mirror(img_A)
        #     img_B = ImageOps.mirror(img_B)
        #     label = ImageOps.mirror(label)   

        if self.random_flip:
            flip_prob = random.random()
            if flip_prob < 0.25:
                pass
            elif flip_prob < 0.5:
                # 水平翻转
                img_A = ImageOps.mirror(img_A)
                img_B = ImageOps.mirror(img_B)
                label = ImageOps.mirror(label)
            elif flip_prob < 0.75:
                # 垂直翻转
                img_A = ImageOps.flip(img_A)
                img_B = ImageOps.flip(img_B)
                label = ImageOps.flip(label)
            else:
                # 水平+垂直翻转
                img_A = ImageOps.mirror(img_A)
                img_B = ImageOps.mirror(img_B)
                label = ImageOps.mirror(label)
                img_A = ImageOps.flip(img_A)
                img_B = ImageOps.flip(img_B)
                label = ImageOps.flip(label)

        out["img_A"] = ( self.pil_to_tensor(img_A).float()/255 - 0.5 ) / 0.5
        out["img_B"] = ( self.pil_to_tensor(img_B).float()/255 - 0.5 ) / 0.5
        out['label'] = ( self.pil_to_tensor(label).float()/255 - 0.5 ) / 0.5
        out['mask'] = torch.tensor(1.0) 

        out['caption'] = 'A remote sensing photo.'

        return out