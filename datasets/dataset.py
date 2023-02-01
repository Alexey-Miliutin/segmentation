import os
import cv2
import torch
import numpy as np
import torch.utils.data as data


def get_img_paths(path: str, mode: str) -> list:
    paths = []
    for item in os.listdir(os.path.join(path, mode, 'images')):
        paths.append(os.path.join(path, mode, 'images', item))
    return paths


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


class Dataset(data.Dataset):

    def __init__(self, opt, mode):
        self.img_paths = get_img_paths(opt.data, mode)
        print('Number of', mode, 'images: {},'.format(len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(img_path.replace('images', 'masks'), 0)

        return to_float_tensor(img) / 255, torch.from_numpy(np.expand_dims(mask, 0)) / 255


class DataLoader(data.DataLoader):

    def __init__(self, opt, mode) -> None:
        self.opt = opt
        self.dataset = Dataset(opt, mode)
        self.dataloader = data.DataLoader(
        dataset=self.dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_workers)
  
    def load_data(self):
        return self
  
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for data in self.dataloader:
            yield data