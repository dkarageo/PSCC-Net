import os
import pathlib

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
    output_dir: pathlib.Path
) -> None:
    images: list[torch.Tensor] = torch.split(image, 1, dim=0)
    batch_num: int = len(images)

    output_dir.mkdir(exist_ok=True, parents=True)

    for ind in range(batch_num):
        save_path: pathlib.Path = output_dir / f'{pathlib.Path(image_name[ind]).stem}.png'
        tv_utils.save_image(images[ind], save_path)


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