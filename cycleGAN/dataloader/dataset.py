import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
torch.manual_seed(0)

class StyleTransferDataset(Dataset):
    def __init__(self, content_folder="faces/final/", style_folder="cubism/", transform=None):
        self.transform = transform
        self.content_files = sorted(glob.glob(os.path.join(content_folder, "*.jpg")))
        self.style_files = sorted(glob.glob(os.path.join(style_folder, "*")))
        self.new_perm()

    def new_perm(self):
        if len(self.content_files) > len(self.style_files):
            all_style_indices = list(range(len(self.style_files)))
            random.shuffle(all_style_indices)
            self.randperm = torch.tensor(all_style_indices * (len(self.content_files) // len(all_style_indices) + 1))[:len(self.content_files)]
        else:
            self.randperm = torch.randperm(len(self.style_files))[:len(self.content_files)]

    def __getitem__(self, index):
        content_item = self.transform(Image.open(self.content_files[index % len(self.content_files)]))
        style_item = self.transform(Image.open(self.style_files[self.randperm[index]]))

        if content_item.shape[0] != 3:
            content_item = self.adjust_channels(content_item, 3)
        if style_item.shape[0] != 3:
            style_item = self.adjust_channels(style_item, 3)

        if index == len(self) - 1:
            self.new_perm()

        return (content_item - 0.5) * 2, (style_item - 0.5) * 2

    def __len__(self):
        return len(self.content_files) if len(self.content_files) > len(self.style_files) else len(self.style_files)

    def adjust_channels(self, img, target_channels):
        img_channels, _, _ = img.shape
        if img_channels < target_channels:
            img = img.repeat(target_channels // img_channels, 1, 1)
        elif img_channels > target_channels:
            img = img[:target_channels, :, :]
        return img