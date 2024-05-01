import csv
import pathlib
import timeit
from typing import Any, Optional

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import save_image
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData
from utils import csv_utils

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda:0')
# device = torch.device('cpu')


@click.command()
@click.option("--input_dir",
              type=click.Path(exists=True, path_type=pathlib.Path),
              default="./sample",
              help="Directory where input images are located.")
@click.option("--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              default="./mask_results",
              help="Directory where output masks will be written.")
@click.option("--csv_root_dir", "-r",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Directory to which paths into the provided input CSV file are relative to. "
                   "This option is only meaningful when the `input_dir` option points to a CSV "
                   "file.")
@click.option("--update_input_csv", is_flag=True, default=False,
              help="When a CSV file is provided as input, providing this flag enables you to "
                   "update the input CSV, instead of exporting a new one.")
def cli(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    csv_root_dir: Optional[pathlib.Path],
    update_input_csv: bool
) -> None:
    args = get_pscc_args()
    test(args, input_dir, output_dir,
         csv_root_dir=csv_root_dir,
         update_input_csv=update_input_csv)


def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cuda:0')
    # net_state_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(net_state_dict)
    print('{} weight-loading succeeds'.format(name))


def test(
    args,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    csv_root_dir: Optional[pathlib.Path],
    update_input_csv: bool
) -> None:
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)

    # load FENet weight
    FENet = FENet.to(device)
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    test_data = TestData(args, input_dir, csv_root_dir=csv_root_dir)
    test_data_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    detection_results: list[dict[str, Any]] = []
    localization_masks: dict[pathlib.Path, pathlib.Path] = {}
    detection_scores: dict[pathlib.Path, float] = {}

    torch.cuda.synchronize()
    start_time: float = timeit.default_timer()

    for batch_id, test_data in tqdm(enumerate(test_data_loader),
                                    desc="Analyzing images",
                                    unit="image",
                                    total=len(test_data)):

        image, cls, name = test_data
        image = image.to(device)

        with torch.no_grad():

            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)),
                                      mode='bilinear', align_corners=True)

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

        if args.save_tag:
            detection_results.append({
                "image": pathlib.Path(name[0]).name,
                "psccnet_detection": pred_logit[0, 1].detach().cpu().item()
            })
            batch_loc_masks: dict[pathlib.Path, pathlib.Path] = save_image(
                pred_mask, name, output_dir, output_relative_to=csv_root_dir
            )
            localization_masks.update(batch_loc_masks)
            detection_scores[pathlib.Path(name[0])] = pred_logit[0, 1].detach().cpu().item()

        # print_name = name[0].split('/')[-1].split('.')[0]
        # print(f'The image {print_name} is {pred_tag}')

        # Clear PyTorch cache for the next sample.
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    stop_time: float = timeit.default_timer()
    elapsed_time: float = stop_time - start_time
    print(f"Total time: {elapsed_time} secs")
    print(f"Time per image: {elapsed_time / len(test_data)}")

    if args.save_tag:
        write_csv_file(detection_results, output_dir/"detection_results.csv")

    if input_dir.is_file() and update_input_csv:
        out_data: csv_utils.AlgorithmOutputData = csv_utils.AlgorithmOutputData(
            tampered=localization_masks,
            untampered=None,
            response_times=None,
            image_level_predictions=detection_scores
        )
        out_data.save_csv(input_dir, csv_root_dir, output_column="psccnet")


def write_csv_file(
    data: list[dict[str, Any]],
    output_file: pathlib.Path,
    fieldnames: Optional[list[str]] = None
) -> None:
    with output_file.open("w") as f:
        if not fieldnames:
            fieldnames = data[0].keys()
        writer: csv.DictWriter = csv.DictWriter(f,
                                                fieldnames=fieldnames,
                                                delimiter=",",
                                                extrasaction="ignore")
        writer.writeheader()
        for r in data:
            writer.writerow(r)


if __name__ == '__main__':
    cli()
