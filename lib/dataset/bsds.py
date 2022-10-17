import os, glob
import torch
import numpy as np
from skimage.feature import local_binary_pattern
import cv2


def convert_label(label, n_class):

    onehot = np.zeros((1, n_class, label.shape[0], label.shape[1])).astype(np.float32)

    for t in np.unique(label).tolist():
        onehot[:, t, :, :] = (label == t)

    return onehot


class BSDS:
    def __init__(self, root, n_class, split="train", color_transforms=None, geo_transforms=None):
        self.gt_dir = os.path.join(root, "ann_dir", split)
        self.img_dir = os.path.join(root, "img_dir", split)
        self.n_class = n_class
        self.index = os.listdir(self.gt_dir)

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms


    def __getitem__(self, idx):
        idx = self.index[idx][:-4]

        gt = cv2.imread(os.path.join(self.gt_dir, idx+".png"), cv2.IMREAD_UNCHANGED)

        img = cv2.imread(os.path.join(self.img_dir, idx+".tiff"), cv2.IMREAD_UNCHANGED)

        gt = gt.astype(np.int64)
        img = img.astype(np.float32)

        if self.color_transforms is not None:
            img = self.color_transforms(img)

        if self.geo_transforms is not None:
            img, gt = self.geo_transforms([img, gt])

        gt = convert_label(gt, self.n_class)
        gt = torch.from_numpy(gt)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        #+++++
        img = img[0, :, :].unsqueeze(0)
        #+++++

        return img, gt.squeeze(0)


    def __len__(self):
        return len(self.index)


class DSSG:
    def __init__(self, root, n_class, split="train", color_transforms=None, geo_transforms=None, n_points=24, radius=3):
        self.gt_dir = os.path.join(root, "ann_dir", split)
        self.img_dir = os.path.join(root, "img_dir", split)
        self.n_class = n_class
        self.index = os.listdir(self.gt_dir)

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms
        self.n_points = n_points
        self.radius = radius


    def __getitem__(self, idx):
        idx = self.index[idx][:-4]

        gt = cv2.imread(os.path.join(self.gt_dir, idx+".png"), cv2.IMREAD_UNCHANGED)

        img = cv2.imread(os.path.join(self.img_dir, idx+".tiff"), cv2.IMREAD_UNCHANGED)

        # lgrp = local_binary_pattern(img[:, :, 0], self.n_points, self.radius, 'var')
        # lgrp = (lgrp / self.n_points) - (lgrp / (self.n_points * self.n_points))
        lgrp = local_binary_pattern(img[:, :, 0], self.n_points, self.radius)


        gt = gt.astype(np.int64)
        img = img.astype(np.float32)

        if self.color_transforms is not None:
            img = self.color_transforms(img)

        if self.geo_transforms is not None:
            img, gt = self.geo_transforms([img, gt])

        gt = convert_label(gt, self.n_class)
        gt = torch.from_numpy(gt)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        img = img[0, :, :].unsqueeze(0)
        lgrp = torch.from_numpy(lgrp)
        return img, gt.squeeze(0), lgrp


    def __len__(self):
        return len(self.index)

















