import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# import segment.val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

from utils.loss_rtdetr import RTDETRSegmentLoss
import segment.val_detr as validate_detr

def validate_only(opt, device):
    # Directories
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # Load hyperparameters
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Load dataset info
    data_dict = check_file(opt.data)  # Kiểm tra file data.yaml
    with open(data_dict, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    val_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # Số lượng lớp
    # names = {0: 'item'} if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    
    cfg = check_yaml(opt.cfg)  # Kiểm tra file .yaml
    model = SegmentationModel(cfg, ch=3, nc=nc).to(device)  # Khởi tạo model từ cfg
    LOGGER.info(f'Model loaded from {cfg}')

    if opt.weights:
        weights = check_file(opt.weights)  # Kiểm tra file weights
        ckpt = torch.load(weights, map_location=device)  # Load checkpoint
        # Lấy state_dict từ checkpoint
        csd = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
        # Giao các key giữa state_dict của checkpoint và model
        csd = intersect_dicts(csd, model.state_dict())  # Loại bỏ các key không khớp
        model.load_state_dict(csd, strict=False)  # Load state_dict vào model
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        LOGGER.info('No weights provided, using random initialized weights from cfg.')

    # Chuyển model sang chế độ evaluation
    model.eval()

    # Check AMP (Automatic Mixed Precision)
    amp = torch.cuda.is_available() and device.type != 'cpu'

    # Dataloader cho validation
    from utils.segment.dataloaders import create_dataloader
    val_loader = create_dataloader(val_path,
                                       opt.imgsz,
                                       opt.batch_size,
                                       32,
                                       opt.single_cls,
                                       hyp=hyp,
                                       cache= opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=opt.workers * 2,
                                       pad=0.5,
                                       mask_downsample_ratio=opt.mask_ratio,
                                       overlap_mask=not opt.no_overlap,
                                       prefix=colorstr('val: '))[0]

    # Validation
    LOGGER.info(f'Starting validation with weights {weights}...')
    results, maps, _ = validate_detr.run(
        data_dict,
        batch_size=opt.batch_size * 2,
        imgsz=opt.imgsz,
        half=amp,  # Sử dụng half-precision nếu AMP khả dụng
        model=model,
        single_cls=opt.single_cls,
        dataloader=val_loader,
        save_dir=save_dir,
        plots=not opt.noplots,
        compute_loss= RTDETRSegmentLoss(nc, use_vfl=True),  # Không cần compute loss khi chỉ validate
        mask_downsample_ratio=opt.mask_ratio,
        overlap=not opt.no_overlap
    )

    # Log kết quả
    LOGGER.info(f'Validation results: {results}')
    return results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="weights/gelan-c-seg.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/segment/gelan-c-seg-detr.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=60, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=160, help='validation image size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cache', action='store_true', help='cache images in RAM')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to save memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')
    return parser.parse_args()

def main(opt):
    # Chọn device
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Chạy validation
    validate_only(opt, device)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)