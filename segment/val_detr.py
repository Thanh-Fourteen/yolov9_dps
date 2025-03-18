import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           non_max_suppression, print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_images
from utils.segment.plots import plot_images_and_masks
from utils.segment.dataloaders import create_dataloader as create_segment_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_upsample, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.torch_utils import de_parallel, select_device, smart_inference_mode
from utils.loss_rtdetr import RTDETRSegmentLoss, HungarianMatcher

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map, pred_masks=None):
    from pycocotools.mask import encode
    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    entry = {
        'image_id': image_id,
        'category_id': class_map[int(predn[0, 5])],
        'bbox': [round(x, 3) for x in box[0].tolist()],
        'score': round(predn[0, 4], 5)
    }
    if pred_masks is not None:
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        entry['segmentation'] = rles[0]
    jdict.append(entry)

def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, is_detr=False):
    correct_bboxes = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_masks = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool) if pred_masks is not None else None
    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct_bboxes[matches[:, 1].astype(int), i] = True
    
    if pred_masks is not None and gt_masks is not None:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou_mask = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        for i in range(len(iouv)):
            x = torch.where((iou_mask >= iouv[i]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou_mask[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct_masks[matches[:, 1].astype(int), i] = True
    
    return (torch.tensor(correct_bboxes, dtype=torch.bool, device=iouv.device),
            torch.tensor(correct_masks, dtype=torch.bool, device=iouv.device) if correct_masks is not None else None)

@smart_inference_mode()
def run(
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
    task='val',
    device='',
    workers=8,
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    project=ROOT / 'runs/val-seg',
    name='exp',
    exist_ok=False,
    half=True,
    dnn=False,
    model=None,
    dataloader=None,
    save_dir=Path(''),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
    is_detr=True  # Thêm tham số để chọn giữa YOLO và DETR
):
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
        nm = de_parallel(model).model[-1].nm  # number of masks
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    model.eval()
    cuda = device.type != 'cpu'
    data = check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '),
                                       overlap_mask=overlap,
                                       mask_downsample_ratio=mask_downsample_ratio)[0]
        
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if data.get('val', '').endswith('val2017.txt') else list(range(1000))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R", "mAP50", "mAP50-95)")
    dt = Profile(), Profile(), Profile()
    metrics = Metrics()
    mloss = torch.zeros(5 if is_detr else 4, device=device)  # GIoU, class, bbox, mask, dice (DETR) hoặc box, obj, cls (YOLO)
    jdict, stats = [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)

    compute_loss = RTDETRSegmentLoss(nc=nc) if is_detr and compute_loss is None else compute_loss

    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            im = im.half() if half else im.float() / 255
            nb, _, height, width = im.shape

        batch_idx = targets[:, 0]
        gt_groups = [(batch_idx == i).sum().item() for i in range(nb)]
        _targets = {
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:],
            "batch_idx": batch_idx.long(),
            "gt_groups": gt_groups,
            "mask": masks
        }

        with dt[1]:
            if is_detr:
                preds = model(im, batch=_targets, detr=True)
                dec_bboxes, dec_scores, dec_masks, enc_bboxes, enc_scores, enc_masks, dn_meta = preds[1]
                pred_bboxes, pred_scores, pred_masks = dec_bboxes[-1], dec_scores[-1], dec_masks[-1]
                
                if dn_meta is not None:
                    dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                    dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
                    dn_masks, dec_masks = torch.split(dec_masks, dn_meta["dn_num_split"], dim=2)
                else:
                    dn_bboxes, dn_scores, dn_masks = None, None, None

                dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
                dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
                dec_masks = torch.cat([enc_masks.unsqueeze(0), dec_masks])

                if compute_loss:
                    loss_dict = compute_loss((dec_bboxes, dec_scores, dec_masks), _targets,
                                            dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_masks=dn_masks, dn_meta=dn_meta)
                    loss_items = torch.as_tensor([loss_dict[k] for k in ["loss_giou", "loss_class", "loss_bbox", "loss_mask", "loss_dice"]], device=device)
                    mloss = (mloss * batch_i + loss_items) / (batch_i + 1)

                bs, _, nd = pred_bboxes.shape
                bboxes, scores = pred_bboxes.split((4, nd - 4), dim=-1)
                topk_values, topk_indexes = torch.topk(scores.reshape(bs, -1), max_det, dim=1)
                topk_boxes = topk_indexes // scores.shape[2]
                lbs = topk_indexes % scores.shape[2]
                bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
                scores = topk_values
                preds = [torch.cat([xywh2xyxy(bbox), score[..., None], cls[..., None]], dim=-1) 
                         for bbox, score, cls in zip(bboxes, scores, lbs)]
            else:  # YOLO
                preds, train_out = model(im)
                protos = train_out[-1]
                if compute_loss:
                    loss_items = compute_loss(train_out, targets, masks)[1]
                    mloss = (mloss * batch_i + loss_items) / (batch_i + 1)
                targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
                preds = non_max_suppression(preds, conf_thres, iou_thres, max_det=max_det, nm=nm)

        with dt[2]:
            plot_masks = []
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]
                path, shape = Path(paths[si]), shapes[si][0]
                correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)
                correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device) if is_detr or protos is not None else None
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct_bboxes, correct_masks, *torch.zeros((2, 0), device=device), labels[:, 0]))
                        if plots:
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue

                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
                gt_masks = masks[targets[:, 0] == si]
                pred_masks = None

                if is_detr:
                    pred_masks = pred_masks[si]
                elif protos is not None:
                    pred_masks = process_mask(protos[si], pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5]) * torch.tensor(im[si].shape[1:], device=device)[[1, 0, 1, 0]]
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct_bboxes, correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap, is_detr)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)

                stats.append((correct_bboxes, correct_masks, pred[:, 4], pred[:, 5], labels[:, 0]))
                
                if pred_masks is not None and plots and batch_i < 3:
                    plot_masks.append(pred_masks[:15].cpu())

                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
                if save_json:
                    pred_masks_scaled = scale_image(im[si].shape[1:], pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), 
                                                  shape, shapes[si][1]) if pred_masks is not None else None
                    save_one_json(predn, jdict, path, class_map, pred_masks_scaled)

        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    t = tuple(x.t / seen * 1E3 for x in dt)
    if not training:
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(batch_size, 3, imgsz, imgsz)}' % t)

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")
        LOGGER.info(f'\nSaving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()
    return (*metrics.mean_results(), *(mloss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo-seg.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--is-detr', action='store_true', help='use DETR/RT-DETR instead of YOLO')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt

def main(opt):
    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)