import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # DEYO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, print_args, scale_boxes,
                           xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images
from utils.torch_utils import select_device, smart_inference_mode
from utils.segment.general import mask_iou, process_mask_upsample, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map, pred_masks):
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))  # (h,w,n) -> (n,h,w)
    rles = [single_encode(m) for m in pred_masks]
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]})

def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, masks=False):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    if masks:
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:
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
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

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
        compute_loss=None,
        callbacks=Callbacks()
):
    if save_json:
        check_requirements(['pycocotools'])

    # Initialize/load model and set device
    training = model is not None
    if training:
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        half &= device.type != 'cpu'
        model.half() if half else model.float()
    else:
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)
        half = model.fp16
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')
        data = check_dataset(data)

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), detr=True)
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=0.5,
                                       rect=pt,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]  # Update to return masks

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)")
    metrics = Metrics()
    mloss = torch.zeros(3, device=device)
    jdict, stats = [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)

    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        with Profile() as dt0:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            im = im.half() if half else im.float()
            im /= 255
            masks = masks.float()  # Ensure masks are float for processing
            nb, _, height, width = im.shape

        # Inference
        with Profile() as dt1:
            preds = model(im, batch=None, detr=True)  # No batch info needed for pure validation
            pred_masks = preds[0][1] if len(preds[0]) > 1 else None  # Assuming masks are in preds[0][1]
            preds = preds[0][0] if len(preds[0]) > 1 else preds[0]  # Bbox and scores

            if compute_loss:
                dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds[1]
                if dn_meta is None:
                    dn_bboxes, dn_scores, dn_masks = None, None, None
                else:
                    dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                    dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
                    dn_masks = pred_masks[:dn_meta["dn_num_split"]] if pred_masks is not None else None
                    pred_masks = pred_masks[dn_meta["dn_num_split"]:] if pred_masks is not None else None
                dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
                dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
                batch = {
                    "cls": targets[:, 1].long(),
                    "bboxes": targets[:, 2:],
                    "gt_groups": [(targets[:, 0] == i).sum().item() for i in range(nb)],
                    "mask": masks  # Use ground truth masks from dataloader
                }
                loss = compute_loss((dec_bboxes, dec_scores, pred_masks), batch, dn_bboxes, dn_scores, dn_masks, dn_meta)
                loss, loss_items = sum(loss.values()), torch.as_tensor(
                    [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=device)
                mloss = (mloss * batch_i + loss_items) / (batch_i + 1)

        # Post-processing
        bs, _, nd = preds.shape
        bboxes, scores = preds.split((4, nd - 4), dim=-1)
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        topk_values, topk_indexes = torch.topk(scores.reshape(bs, -1), max_det, dim=1)
        topk_boxes = topk_indexes // scores.shape[2]
        lbs = topk_indexes % scores.shape[2]
        bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        scores = topk_values

        for i, (bbox, score, cls) in enumerate(zip(bboxes, scores, lbs)):
            bbox = xywh2xyxy(bbox)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred

        # Metrics
        for si, pred in enumerate(outputs):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(None, labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            # Masks
            if pred_masks is not None:
                pred_masks_si = pred_masks[si]  # Shape: (num_queries, h, w)
                gt_masks = masks[si].unsqueeze(0) if masks[si].dim() == 2 else masks[si]  # Ensure (n, h, w)
                pred_masks_si = F.interpolate(pred_masks_si[None], size=gt_masks.shape[1:], mode="bilinear")[0]
                pred_masks_si = pred_masks_si.sigmoid() > 0.5  # Binarize masks
                pred_masks_si = pred_masks_si[:npr]  # Match number of predictions

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.tensor(im[si].shape[1:], device=device)[[1, 0, 1, 0]]
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct_bboxes = process_batch(predn, labelsn, iouv)
                if pred_masks is not None and nl > 0:
                    correct_masks = process_batch(predn, labelsn, iouv, pred_masks_si, gt_masks, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json and pred_masks is not None:
                pred_masks_scaled = scale_image(im[si].shape[1:], pred_masks_si.cpu().numpy(), shape, shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks_scaled)

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            if pred_masks is not None:
                plot_images_and_masks(im, output_to_target(outputs, max_det=15), pred_masks_si[:15], paths,
                                      save_dir / f'val_batch{batch_i}_pred.jpg', names)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    LOGGER.info(f"Val/loss: {mloss.cpu().tolist()}")
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set')

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            results = []
            for eval in COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm'):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = metrics.mean_results()
    return (*final_metric, *(mloss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), (dt0.t, dt1.t, 0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'deyo-seg.pt', help='model path(s)')
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
    parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt

def main(opt):
    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)