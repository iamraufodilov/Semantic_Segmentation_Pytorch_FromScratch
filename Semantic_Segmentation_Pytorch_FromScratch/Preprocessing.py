# import libraries

#base
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

#pytorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler

# data split libraries
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

#augementation libraries
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor

# other libraries
import os
import pdb
import time 
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures

warnings.filterwarnings("ignore")

# fixing seed
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# data preprocessing

# to change gif image to png
PATH = 'G:/rauf/STEPBYSTEP/Data/Carvana(Kaggle)/'
(PATH/'train_masks_png').mkdir(exist_ok=True)
def conver_img(fn):
    fn = fn.name
    PIL.Image.open(PATH/'train_maks'/fn).save(PATH/'train_masks_png'/f'{fn[:-4]}.png')
files = list((PATH/'train_masks').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(convert_img, files)

# to chnge img mask size
(PATH/'train_masks-128').mkdir(exist_ok=True)
def resize_mask(fn):
    PIL.Image.open(fn).resize((128, 128)).save((fn.parent.parent)/'train_masks-128'/ fn.name)

files = list((PATH/'train_masks_png').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_mask, files)

# to change image size
(PATH/'train-128').mkdir(exist_ok=True)
def resize_img(fn):
    PIL.Image.open(fn).resize((128, 128)).save((fn.parent.parent)/'train-128'/ fn.name)

files = list((PATH/'train').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_img, files)


df=pd.read_csv('G:/rauf/STEPBYSTEP/Data/Carvana(Kaggle)/')
# location of original and mask image
img_fol='G:/rauf/STEPBYSTEP/Data/Carvana(Kaggle)/train-128'
mask_fol='G:/rauf/STEPBYSTEP/Data/Carvana(Kaggle)/train_masks-128'
# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)

# data loading pipeline

# for training phase
def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean,std=std, p=1), ToTensor()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans

class CarDataset(Dataset):
    def __init__(self,df,img_fol,mask_fol,mean,std,phase):
        self.fname=df['img'].values.tolist()
        self.img_fol=img_fol
        self.mask_fol=mask_fol
        self.mean=mean
        self.std=std
        self.phase=phase
        self.trasnform=get_transform(phase,mean,std)
    def __getitem__(self, idx):
        name=self.fname[idx]
        img_name_path=os.path.join(self.img_fol,name)
        mask_name_path=img_name_path.split('.')[0].replace('train-128','train_masks-128')+'_mask.png'
        img=cv2.imread(img_name_path)
        mask=cv2.imread(mask_name_path,cv2.IMREAD_GRAYSCALE)
        augmentation=self.trasnform(image=img, mask=mask)
        img_aug=augmentation['image']                           #[3,128,128] type:Tensor
        mask_aug=augmentation['mask']                           #[1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)

# split dat for training and validation
def CarDataloader(df,img_fol,mask_fol,mean,std,phase,batch_size,num_workers):
    df_train,df_valid=train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase=='train' else df_valid
    for_loader=CarDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader=DataLoader(for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return dataloader