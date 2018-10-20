import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from .transform import *

IM_SIZE = 101


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def normalize(im):
    max = np.max(im)
    min = np.min(im)
    if (max - min) > 0:
        im = (im - min) / (max - min)
    return im


def train_augment(image, mask, index):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)  # 0.125

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
            pass

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))  # 10

        if c == 3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.15))  # 0.10
            pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        # if c==1:
        #     image = do_invert_intensity(image)

    return image, mask, index

def add_depth_channels(image_tensor):
        _, h, w = image_tensor.size()
        for row, const in enumerate(np.linspace(0, 1, h)):
            image_tensor[1, row, :] = const
        image_tensor[2] = image_tensor[0] * image_tensor[1]
        return image_tensor
    
class TorchDataset(Dataset):

    def __init__(self, df, is_test=False, transform=None):
        self.df = df
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pad = ((0, 0), (14, 13), (14, 13))

        im = self.df.images.iloc[index]

        if not self.is_test:
            mask = self.df.masks.iloc[index]
            if not self.is_test:
                # no-empty
                if np.sum(mask.reshape(101, 101)) > 0.0:
                    target = [1]
                # empty
                else:
                    target = [0]

            if self.transform is not None:
                im, mask, index = self.transform(im, mask, index)

            # 在0 axis那里增加一维度，(1 , 101, 101)
            mask = np.expand_dims(mask, 0)
            # ‘edge’——表示用边缘值填充
            mask = np.pad(mask, pad, 'edge')
            mask = torch.from_numpy(mask).float()

        if len(im.shape) == 2:
            depth = np.ones_like(im) * self.df.z.iloc[index]
            im = np.stack([im, depth, depth], axis=0)
        elif len(im.shape) == 3:
            im = np.rollaxis(im, 2, 0)
        # im = np.expand_dims(im, 0)
        im = np.pad(im, pad, 'edge')
        im = torch.from_numpy(im).float()
        # im = add_depth_channels(im)
        z = torch.from_numpy(np.expand_dims(self.df.z.iloc[index], 0)).float()
        if not self.is_test:
            target = torch.from_numpy(np.array(target)).float()
        if self.is_test:
            return self.df.id.iloc[index], im, 
        else:
            return self.df.id.iloc[index], im, mask,  target


class TGS_Dataset():

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df = self.create_dataset_df(self.folder_path)
        self.df['z'] = normalize(self.df['z'].values)
        try:
            empty = np.array([np.sum(m) for m in self.df.masks])
            print('{} empty masks out of {} total masks'.format(np.sum(empty == 0), len(empty)))
        except AttributeError:
            pass

    @staticmethod
    def load_images(df, data='train'):
        df['images'] = [
            normalize(cv2.imread(df.iloc[i]['im_path'],
                       cv2.IMREAD_COLOR).astype(np.float32)) for i in range(len(df))]
        if data == 'train':
            df['masks'] = [
                normalize(cv2.imread(df.iloc[i]['mask_path'],
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)) for i in range(len(df))]
        return df

    @staticmethod
    def create_dataset_df(folder_path, load=True):
        '''Create a dataset for a specific dataset folder path'''
        # Walk and get paths
        walk = os.walk(folder_path)
        main_dir_path, subdirs_path, csv_path = next(walk)
        dir_im_path, _, im_path = next(walk)
        # Create dataframe
        df = pd.DataFrame()
        df['id'] = [im_p.split('.')[0] for im_p in im_path]
        df['im_path'] = [os.path.join(dir_im_path, im_p) for im_p in im_path]
        if any(['mask' in sub for sub in subdirs_path]):
            data = 'train'
            dir_mask_path, _, mask_path = next(walk)
            df['mask_path'] = [os.path.join(dir_mask_path, m_p)
                               for m_p in mask_path]
            rle_df = pd.read_csv(os.path.join(main_dir_path, csv_path[1]))
            df = df.merge(rle_df, on='id', how='left')
        else:
            data = 'test'

        depth_df = pd.read_csv(os.path.join(main_dir_path, csv_path[0]))
        df = df.merge(depth_df, on='id', how='left')

        if load:
            df = TGS_Dataset.load_images(df, data=data)
        return df

    def yield_dataloader(self, data='train', nfold=5,
                         shuffle=True, seed=143, stratify=True,
                         num_workers=8, batch_size=10, auxiliary_df=None):

        if data == 'train':
            if stratify:
                self.df["coverage"] = self.df.masks.map(np.sum) / pow(IM_SIZE, 2)
                self.df["coverage_class"] = self.df.coverage.map(cov_to_class)
                kf = StratifiedKFold(n_splits=nfold,
                                     shuffle=True,
                                     random_state=seed)
            else:
                kf = KFold(n_splits=nfold,
                           shuffle=True,
                           random_state=seed)
            loaders = []
            idx = []
            self.df = self.df.fillna(0.0)
            for train_ids, val_ids in kf.split(self.df['id'].values, self.df.coverage_class):
                if auxiliary_df is not None:
                    train_df = self.df.iloc[train_ids].append(auxiliary_df)
                else:
                    train_df = self.df.iloc[train_ids]

                train_dataset = TorchDataset(train_df,
                                             transform=train_augment)
                train_loader = DataLoader(train_dataset,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          batch_size=batch_size,
                                          pin_memory=True)

                val_dataset = TorchDataset(self.df.iloc[val_ids])
                val_loader = DataLoader(val_dataset,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        pin_memory=True)
                idx.append((self.df.id.iloc[train_ids], self.df.id.iloc[val_ids]))
                loaders.append((train_loader, val_loader))
            return loaders, idx

        elif data == 'test':
            test_dataset = TorchDataset(self.df, is_test=True)
            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     batch_size=batch_size,
                                     pin_memory=True)
            return test_loader, self.df.id
