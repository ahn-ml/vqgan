import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex



class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

from torch.utils.data import Dataset
import torch
from typing import Optional, Callable
import os

from torchvision import transforms
import argparse
import h5pickle as h5py
import numpy as np
from einops import rearrange
from torchvision.transforms import functional as F

class DancingSpritesDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        img_size: int,
        video_len: int,
        num_train_images: Optional[int] = None,
        transform: Optional[Callable] = None, 
        stochastic_sample: Optional[bool] = True,
        include_labels: Optional[bool] = False,
    ):
        self.img_size = img_size
        self.video_len = video_len
        self.split = split
        self.stochastic_sample = stochastic_sample
        self.include_labels = include_labels
        
        if split == 'train':
            dataset = h5py.File(f'{root}/train.h5','r')
            self.images = dataset[split]['images']
            if self.include_labels:
                self.labels = dataset[split]['labels']
        elif split == 'test':
            dataset = h5py.File(f'{root}/test.h5','r')
            self.images = dataset[split]['images']
            self.labels = dataset[split]['labels']
            #self.masks = dataset[split]['masks']
        
        if num_train_images:
            inds = np.sort((torch.randperm(len(self.images))).cuda()[:num_train_images].cpu().numpy())
            
            self.images = self.images[inds]
            if self.split =="test":
                self.labels = self.labels[inds]
                #self.masks = self.masks[inds]
            
        print(f'{split} dataset has # {len(self.images)}')
        
        # normalize = transforms.Compose([
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        normalize = transforms.Compose([
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        
        self.transform = normalize if transform is None else transform
        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        # stochastic video sample
        if self.stochastic_sample:
            start_idx = torch.randperm(len(self.images[idx]) - self.video_len + 1)[0]
        else:
            start_idx = 0
        
        # transform PIL -> normalized tensor
        transformed_image=self.images[idx][start_idx:start_idx+self.video_len]
        transformed_image=F.resize(torch.tensor(rearrange(transformed_image, 'f h w c -> f c h w')/255.0), self.img_size)
        mask = torch.zeros_like(transformed_image)
        if self.split =="test" or self.include_labels:
            label = self.labels[idx][start_idx:start_idx+self.video_len]
        if self.split == "test":
            #mask = F.resize(torch.tensor(self.masks[idx][start_idx:start_idx+self.video_len]).squeeze(), self.img_size)
            pass
        
        source = transformed_image[:self.video_len]
        if self.split == "train" and not self.include_labels:
            return self.transform(source.float()).squeeze()
        else:
            return self.transform(source.float()).squeeze() , label, 0

class DSPTrain(DancingSpritesDataset):
    def __init__(self, size, video_len, root="moving-sprites", num_train_images=None, transform=None, stochastic_sample=True, include_labels=False):
        super().__init__(root=root, split='train', img_size=size, video_len=video_len, num_train_images=num_train_images, transform=transform, stochastic_sample=stochastic_sample, include_labels=include_labels)

class DSPTest(DancingSpritesDataset):
    def __init__(self, size, video_len, root="moving-sprites", num_train_images=None, transform=None, stochastic_sample=True, include_labels=False):
        super().__init__(root=root, split='test', img_size=size, video_len=video_len, num_train_images=num_train_images, transform=transform, stochastic_sample=stochastic_sample, include_labels=include_labels)

    
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
