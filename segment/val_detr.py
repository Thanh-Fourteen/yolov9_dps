import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
# from utils.segment.general import mask_iou
from utils.segment.metrics import Metrics, ap_per_class
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import select_device, smart_inference_mode

def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate IoU between two sets of binary masks.

    Args:
        mask1: [N, n] - Predicted masks, N is number of predicted objects, n is flattened size (w * h)
        mask2: [M, n] - Ground truth masks, M is number of gt objects, n is flattened size (w * h)
        eps: Small value to avoid division by zero

    Returns:
        iou: [N, M] - IoU scores between each pair of predicted and gt masks
    """
    # Đảm bảo mask1 và mask2 là nhị phân (nếu cần)
    mask1 = mask1.float()  # Chuyển sang float nếu chưa
    mask2 = mask2.float()

    # Tính intersection
    intersection = torch.matmul(mask1, mask2.t()).clamp(min=0)

    # Tính union
    area1 = mask1.sum(dim=1, keepdim=True)  # [N, 1]
    area2 = mask2.sum(dim=1, keepdim=False)  # [M]
    union = area1 + area2[None, :] - intersection  # [N, M]

    # Tính IoU
    iou = intersection / (union + eps)
    return iou.clamp(min=0, max=1)  # Giới hạn IoU trong [0, 1]

def ap_per_class_box_and_mask(tp_b, tp_m, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """Tính toán chỉ số AP cho hộp giới hạn và mặt nạ."""
    def compute_ap(tp, conf, pred_cls, target_cls):
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]
        ap = np.zeros((nc, tp.shape[1]))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]
            n_p = i.sum()
            if n_p == 0 or n_l == 0:
                continue
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc + 1e-16)
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap_score(recall[:, j], precision[:, j])
        return ap

    def compute_ap_score(recall, precision):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([1.], precision, [0.]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
        return ap

    # Tính AP cho hộp giới hạn
    results_boxes = ap_per_class(tp_b, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Box")
    tp_b, fp_b, p_b, r_b, f1_b, ap_b, ap_class_b = results_boxes

    # Tính AP cho mặt nạ
    results_masks = ap_per_class(tp_m, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Mask")
    print(f"True positives for masks: {tp_m.sum()}")
    tp_m, fp_m, p_m, r_m, f1_m, ap_m, ap_class_m = results_masks

    results = {
        "boxes": {"p": p_b, "r": r_b, "ap": ap_b, "f1": f1_b, "ap_class": ap_class_b},
        "masks": {"p": p_m, "r": r_m, "ap": ap_m, "f1": f1_m, "ap_class": ap_class_m}
    }
    return results


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]})
    
def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        
        # Nội suy pred_masks lên kích thước của gt_masks
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            pred_masks = F.interpolate(pred_masks[None], gt_masks.shape[1:], mode="bilinear", align_corners=False)[0]
        # if gt_masks.shape[1:] != pred_masks.shape[1:]:
        #     pred_masks = F.interpolate(pred_masks[None], gt_masks.shape[1:], mode="bilinear", align_corners=False)[0]
        
        # Ngưỡng hóa
        pred_masks = (pred_masks > 0.5).float()
        gt_masks = gt_masks.gt_(0.5)
        
        # Debug
        print(f"GT masks shape: {gt_masks.shape}, Pred masks shape: {pred_masks.shape}")
        print(f"GT unique: {torch.unique(gt_masks)}, Pred unique: {torch.unique(pred_masks)}")
        
        # Tính IoU
        gt = gt_masks.view(gt_masks.shape[0], -1)
        pm = pred_masks.view(pred_masks.shape[0], -1)
        iou = mask_iou(gt, pm)

        print(f"Mask IoU: {iou}")
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def compute_segmentation_metrics(pred_masks, gt_masks, iou_threshold=0.5):
    """Tính toán IoU và Dice Score cho mặt nạ phân đoạn."""
    pred_masks = (pred_masks > 0.5).float()  # Ngưỡng hóa mặt nạ dự đoán
    intersection = (pred_masks * gt_masks).sum(dim=(-2, -1))  # Giao giữa dự đoán và nhãn
    union = pred_masks.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection  # Hợp
    iou = intersection / (union + 1e-6)  # IoU
    dice = 2 * intersection / (pred_masks.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) + 1e-6)  # Dice Score
    
    # Đếm số lượng mặt nạ đạt ngưỡng IoU
    correct = (iou >= iou_threshold).float()
    return iou.mean(), dice.mean(), correct.mean()

@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val-seg',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        overlap=False,
        mask_downsample_ratio=1,
        compute_loss=None,
        callbacks=Callbacks(),
):
    
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
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} classes)'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), detr=True)
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)
        task = task if task in ('train', 'val', 'test') else 'val'
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
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R", "mAP50", "mAP50-95)")
    dt = Profile(), Profile(), Profile()
    mloss = torch.zeros(5, device=device)
    jdict, stats = [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        print(f"\nBatch {batch_i} - masks type: {type(masks)}")
        print(f"Batch {batch_i} - masks length: {len(masks) if not torch.is_tensor(masks) else masks.numel()}, "
            f"mask shape: {masks[0].shape if len(masks) > 0 else None}")
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = [m.to(device).float() for m in masks]  # Chuyển masks sang device
            im = im.half() if half else im.float()
            im /= 255

        # Chuẩn bị nhãn mặt nạ
        bs = len(im)
        batch_idx = targets[:, 0]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]

        bb = targets[:,2:].to(device)
        cls =  targets[:,1].to(device, dtype=torch.long)
        print(f"\ncls shape: {cls.shape}")
        print(f"\nbb shape: {bb.shape}")
        print(f"\nmasks shape: {masks.shape}")
        print(f"\ngt_groups shape: {len(gt_groups)}")

        _targets = {
            "cls": targets[:,1].to(device, dtype=torch.long),
            "bboxes": targets[:, 2:].to(device),
            "batch_idx": batch_idx.to(device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
            "mask": masks,
        }

        with dt[1]:
            preds = model(im, batch=_targets, detr=True)

            if compute_loss:
                dec_bboxes, dec_scores, dec_masks, enc_bboxes, enc_scores, enc_masks, dn_meta = preds[1]
                print(f"Predicted masks shape: {dec_masks.shape}")
                if dn_meta is None:
                    dn_bboxes, dn_scores, dn_masks = None, None, None
                else:
                    dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                    dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
                    dn_masks, dec_masks = torch.split(dec_masks, dn_meta["dn_num_split"], dim=2)
                    
                dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
                dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
                dec_masks = torch.cat([enc_masks.unsqueeze(0), dec_masks])

                loss = compute_loss((dec_bboxes, dec_scores, dec_masks), _targets,
                                    dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_masks=dn_masks,
                                    dn_meta=dn_meta)
                loss, loss_items = sum(loss.values()), torch.as_tensor(
                    [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox", "loss_mask", "loss_dice"]], device=device
                )
                mloss = (mloss * batch_i + loss_items) / (batch_i + 1)

        # Lọc hộp giới hạn
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        topk_values, topk_indexes = torch.topk(scores.reshape(scores.shape[0], -1), max_det, dim=1)
        topk_boxes = topk_indexes // scores.shape[2]
        lbs = topk_indexes % scores.shape[2]
        bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        scores = topk_values

        for i, bbox in enumerate(bboxes):
            bbox = xywh2xyxy(bbox)
            score = scores[i]
            cls = lbs[i]
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred
        preds = outputs

        # Đánh giá
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_bboxes, correct_masks, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            # Đánh giá phát hiện đối tượng và phân đoạn
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.tensor(im[si].shape[1:], device=device)[[1, 0, 1, 0]]
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                
                # Đánh giá hộp giới hạn
                correct_bboxes = process_batch(predn, labelsn, iouv)
                
                # Đánh giá mặt nạ
                gt_mask = _targets["mask"][si]
                if gt_mask.numel() > 0:
                    pred_masks = dec_masks[-1, si]  # Tầng cuối của dec_masks
                    topk_pred_masks = pred_masks[topk_boxes[si]]
                    print(f"Top-k predicted masks shape: {topk_pred_masks.shape}")
                    correct_masks = process_batch(predn, labelsn, iouv, topk_pred_masks, gt_mask, masks=True)
                    print(f"Sample {si}: {nl} labels, {npr} predictions")
                    print(f"GT mask shape: {gt_mask.shape}, Pred mask shape: {topk_pred_masks.shape}")
                    print(f"Correct masks: {correct_masks.sum()}")
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            
            stats.append((correct_bboxes, correct_masks, pred[:, 4], pred[:, 5], labels[:, 0]))

            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)

            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Tính toán chỉ số
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    print(f"Stats length: {len(stats)}, Correct masks sum: {stats[1].sum()}")
    metrics = Metrics()
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)

    # In kết quả
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8
    LOGGER.info(pf % ('all', seen, nt.sum(), *metrics.mean_results()))
    LOGGER.info(('%22s' + '%11.3g' * 5) % ('val/loss', *(mloss.cpu() / len(dataloader)).tolist()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    if verbose or (nc < 50 and not training) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    t = tuple(x.t / seen * 1E3 for x in dt)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

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
            for eval in (COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm')):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = (mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask,
                    *(mloss.cpu() / len(dataloader)).tolist())
    return final_metric, metrics.get_maps(nc), t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolo.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolo.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)