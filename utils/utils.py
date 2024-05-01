import os
import pathlib
from typing import Optional

import torch
import torchvision.utils as tv_utils


def adjust_learning_rate(optimizer, epoch, lr_strategy, lr_decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    current_learning_rate = lr_strategy[epoch // lr_decay_step]
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))


def save_image(
    image: torch.Tensor,
    image_name: list[str],
    output_dir: pathlib.Path,
    output_relative_to: Optional[pathlib.Path] = None,
) -> dict[pathlib.Path, pathlib.Path]:
    images: list[torch.Tensor] = torch.split(image, 1, dim=0)

    saved_image_paths: dict[pathlib.Path, pathlib.Path] = {}

    for image, image_path in zip(images, image_name):
        if output_relative_to is None:
            image_output_dir: pathlib.Path = output_dir
        else:
            image_output_dir: pathlib.Path = (
                output_dir / pathlib.Path(image_path).parent.relative_to(output_relative_to)
            )
        image_output_dir.mkdir(exist_ok=True, parents=True)
        save_path: pathlib.Path = image_output_dir / f'{pathlib.Path(image_path).stem}.png'
        tv_utils.save_image(image, save_path)
        saved_image_paths[pathlib.Path(image_path)] = save_path

    return saved_image_paths


def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split('.')[0].split('_')[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0