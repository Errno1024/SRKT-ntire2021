import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional
import random

import skimage.io

import os

from tqdm import tqdm

def pwd() -> str:
    return os.path.abspath(os.curdir)

def ext(path: str) -> str:
    rf = path.rfind(os.extsep)
    if rf >= 0:
        return path[rf + 1:]
    return ""

def readimage(file: str) -> torch.Tensor:
    img = skimage.io.imread(file) / 255
    return torch.Tensor(np.transpose(img, (2, 0, 1)))[0:3, ...].to('cpu')

image_transpose = lambda img: img.permute([i for i in range(len(img.shape) - 2)] + [-1, -2])

img_trans = transforms.Compose([
    transforms.RandomApply(transforms.Lambda(image_transpose)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(size=256),
])

IMAGE_EXTENSION = {'png', 'jpg', 'jpeg', 'bmp'}

from collections import Iterable

class PairDataset(Dataset):
    def __init__(self, pairs, random_crop=None):
        self.pairs = pairs
        self.crop = random_crop
        if self.crop is not None:
            if isinstance(self.crop, Iterable):
                it = iter(self.crop)
                w = next(it)
                self.crop = (w, next(it))
            else:
                self.crop = (self.crop, self.crop)

    def __len__(self):
        return len(self.pairs)

    def shuffle(self):
        random.shuffle(self.pairs)

    def __getitem__(self, item):
        if self.crop is not None:
            train, gt, *_ = self.pairs[item]
            h, w = train.shape[-2:]

            crop = self.crop
            transpose = random.random() < 0.5
            if transpose:
                crop = (self.crop[1], self.crop[0])

            y, x = random.randrange(h - crop[1]), random.randrange(w - crop[0])
            train = torchvision.transforms.functional.crop(train, y, x, crop[1], crop[0])
            gt = torchvision.transforms.functional.crop(gt, y, x, crop[1], crop[0])
            if random.random() < 0.5:
                train = transforms.functional.hflip(train)
                gt = transforms.functional.hflip(gt)
            if random.random() < 0.5:
                train = transforms.functional.vflip(train)
                gt = transforms.functional.vflip(gt)
            if transpose:
                train = image_transpose(train)
                gt = image_transpose(gt)
            return (train, gt, *_)
        return self.pairs[item]

def dataset(train_dir='datasets/train', gt_dir='datasets/groundtruth', valid=False, verbose=False, crop=False, cropsize=256):
    curdir = pwd()
    dataset = {}
    gen = None
    try:
        os.chdir(gt_dir)
        gtfiles = set(os.listdir())
        os.chdir(curdir)
        os.chdir(train_dir)
        gen = os.listdir()
        if verbose:
            gen = tqdm(gen, desc='Train set')
        for file in gen:
            if ext(file.lower()) in IMAGE_EXTENSION and file in gtfiles:
                try:
                    img = readimage(file)
                except Exception as e:
                    print(e)
                    continue
                dataset[file] = [img]
    finally:
        os.chdir(curdir)
        if verbose and gen:
            gen.close()
    gen = None
    try:
        os.chdir(gt_dir)
        gen = os.listdir()
        if verbose:
            gen = tqdm(gen, desc='Groundtruth')
        for file in gen:
            if ext(file.lower()) in IMAGE_EXTENSION and file in dataset:
                try:
                    img = readimage(file)
                except Exception as e:
                    print(e)
                    continue
                dataset[file].append(img)
                if valid:
                    dataset[file].append(file)
    finally:
        os.chdir(curdir)
        if verbose and gen:
            gen.close()
    data = []
    for pair in dataset.values():
        if crop:
            if len(pair) == (3 if valid else 2):
                h, w = pair[0].shape[1:]
                if w < 1600 or h < 1200: continue
                train, gt, *_ = pair
                if w > 1600:
                    if h > 1200:
                        train1, gt1 = train[:, :1600, :1200], gt[:, :1600, :1200]
                        train2, gt2 = train[:, -1600:, :1200], gt[:, -1600:, :1200]
                        train3, gt3 = train[:, :1600, -1200:], gt[:, :1600, -1200:]
                        train4, gt4 = train[:, -1600:, -1200:], gt[:, -1600:, -1200:]
                        data.append((train1, gt1, *_))
                        data.append((train2, gt2, *_))
                        data.append((train3, gt3, *_))
                        data.append((train4, gt4, *_))
                    else:
                        train1, gt1 = train[:, :1600, :], gt[:, :1600, :]
                        train2, gt2 = train[:, -1600:, :], gt[:, -1600:, :]
                        data.append((train1, gt1, *_))
                        data.append((train2, gt2, *_))
                elif h > 1200:
                    train1, gt1 = train[:, :, :1200], gt[:, :, :1200]
                    train2, gt2 = train[:, :, -1200:], gt[:, :, -1200:]
                    data.append((train1, gt1, *_))
                    data.append((train2, gt2, *_))
                else:
                    data.append(tuple(pair))
        elif len(pair) == (3 if valid else 2):
            data.append(tuple(pair))
    if valid:
        return PairDataset(data)
    return PairDataset(data, cropsize)

def testset(test_dir='datasets/test', verbose=False):
    curdir = pwd()
    dataset = {}
    gen = None
    try:
        os.chdir(test_dir)
        gen = os.listdir()
        if verbose:
            gen = tqdm(gen, desc='Train set')
        for file in gen:
            if ext(file.lower()) in IMAGE_EXTENSION:
                try:
                    img = readimage(file)
                except Exception as e:
                    print(e)
                    continue
                dataset[file] = img
    finally:
        os.chdir(curdir)
        if verbose and gen:
            gen.close()
    pairs = []
    for file, img in dataset.items():
        pairs.append((img, file))
    return PairDataset(pairs)

def analysis_set(analysis_dir='analysis', base_dir='analysis_base', verbose=True):
    def read(dirname):
        nonlocal verbose

        curdir = pwd()
        dataset = []
        gen = None
        try:
            os.chdir(dirname)
            gen = os.listdir()
            if verbose:
                gen = tqdm(gen, desc='Analysis set')
            for dir in gen:
                if not os.path.isdir(dir): continue
                os.chdir(dir)
                try:
                    images = []
                    for name in ('26.png', '27.png', '28.png', '29.png', '30.png', ):
                        image = readimage(name)
                        images.append(image)
                    with open('scores.txt', 'r') as f:
                        psnr = f.readline()
                        psnr = float(psnr[psnr.rfind(':') + 1:])
                        ssim = f.readline()
                        ssim = float(ssim[ssim.rfind(':') + 1:])
                    dataset.append((dir, (tuple(images), (psnr, ssim))))
                except:
                    continue
                finally:
                    os.chdir('..')
        finally:
            os.chdir(curdir)
            if verbose and gen:
                gen.close()
        return PairDataset(dataset)

    return (read(analysis_dir), read(base_dir))

import torchvision.transforms.functional

def saveimage(image: torch.Tensor, file: str):
    torchvision.transforms.functional.to_pil_image(image).save(file)
