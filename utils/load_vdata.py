import os
import pathlib
from typing import Optional

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image, ImageOps

from utils import csv_utils


class TestData(data.Dataset):
    def __init__(
        self,
        args,
        input_source: pathlib.Path = pathlib.Path('./sample'),
        csv_root_dir: Optional[pathlib.Path] = None
    ):
        super(TestData, self).__init__()

        if input_source.is_dir():
            ddir = str(input_source)
            names = os.listdir(ddir)
            authentic_names = [os.path.join(ddir, name) for name in names if 'authentic' in name]
            fake_names = [os.path.join(ddir, name) for name in names if 'authentic' not in name]
        else:
            manipulated_images: list[pathlib.Path]
            authentic_images: list[pathlib.Path]
            manipulated_images, authentic_images, _ = csv_utils.load_dataset_from_csv(
                input_source, csv_root_dir
            )
            authentic_names = [str(p) for p in authentic_images]
            fake_names = [str(p) for p in manipulated_images]

        authentic_class: list[int] = [0] * len(authentic_names)
        fake_class: list[int] = [1] * len(fake_names)

        self.image_names: list[str] = authentic_names + fake_names
        self.image_class: list[int] = authentic_class + fake_class

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        with Image.open(image_name) as img:
            img.thumbnail((2048, 2048))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = np.array(img)

        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)

        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        return image, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)
