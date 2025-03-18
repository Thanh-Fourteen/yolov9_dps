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
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.metrics import Metrics, ap_per_class
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import select_device, smart_inference_mode

# Import từ file loss_rtdetr.py
from utils.loss_rtdetr import RTDETRSegmentLoss, HungarianMatcher

def mask_iou(mask1, mask2, eps=1e-7):
    """Tính IoU giữa hai tập hợp mặt nạ nhị phân."""
    mask1 = mask1.float()
    mask2 = mask2.float()
    intersection = (mask1 * mask2).sum(dim=(1, 2)).clamp(min=0)
    union = mask1.sum(dim=(1, 2)) + mask2.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + eps)
    return iou.clamp(min=0, max=1)

def ap_per_class_box_and_mask(tp_b, tp_m, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """Tính toán AP cho hộp giới hạn và mặt nạ."""
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

    results_boxes = ap_per_class(tp_b, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Box")
    tp_b, fp_b, p_b, r_b, f1_b, ap_b, ap_class_b = results_boxes

    results_masks = ap_per_class(tp_m, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Mask")
    tp_m, fp_m, p_m, r_m, f1_m, ap_m, ap_class_m = results_masks

    results = {
        "boxes": {"p": p_b, "r": r_b, "ap": ap_b, "f1": f1_b, "ap_class": ap_class_b},
        "masks": {"p": p_m, "r": r_m, "ap": ap_m, "f1": f1_m, "ap_class": ap_class_m}
    }
    return results

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
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
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
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
    """Trả về ma trận dự đoán đúng cho cả bbox và mask."""
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_masks = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)

    # Tính IoU cho bbox
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    # Tính IoU cho mask nếu có
    if masks and pred_masks is not None and gt_masks is not None:
        # Đảm bảo số lượng mask khớp với detections và labels
        n_pred = detections.shape[0]
        n_gt = labels.shape[0]
        if pred_masks.shape[0] > n_pred:
            pred_masks = pred_masks[:n_pred]
        if gt_masks.shape[0] > n_gt:
            gt_masks = gt_masks[:n_gt]

        # Nội suy pred_masks để khớp với kích thước gt_masks
        if pred_masks.shape[1:] != gt_masks.shape[1:]:
            pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=gt_masks.shape[1:], mode="bilinear", align_corners=False).squeeze(0)
        pred_masks = (pred_masks > 0.5).float()
        gt_masks = gt_masks.float()

        # Tính IoU mask
        iou_masks = mask_iou(pred_masks, gt_masks)

    # Xác định dự đoán đúng cho bbox và mask
    for i in range(len(iouv)):
        # Bbox
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

        # Mask
        if masks and pred_masks is not None and gt_masks is not None:
            x_m = torch.where((iou_masks >= iouv[i]) & correct_class.squeeze(-1))
            if x_m[0].shape[0]:
                matches_m = torch.cat((torch.stack(x_m, 1), iou_masks[x_m[0], x_m[1]][:, None]), 1).cpu().numpy()
                if x_m[0].shape[0] > 1:
                    matches_m = matches_m[matches_m[:, 2].argsort()[::-1]]
                    matches_m = matches_m[np.unique(matches_m[:, 1], return_index=True)[1]]
                    matches_m = matches_m[np.unique(matches_m[:, 0], return_index=True)[1]]
                correct_masks[matches_m[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), torch.tensor(correct_masks, dtype=torch.bool, device=iouv.device)

@smart_inference_mode()
def run(
        data,
        weights=None,
        batch_size=32,
        imgsz=640,
        max_det=300,
        task='val',
        device='',
        workers=8,
        single_cls=False,
        verbose=False,
        save_txt=False,
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
):
    # Khởi tạo mô hình và thiết bị
    training = model is not None
    if training:
        device = next(model.parameters()).device
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
            if not (pt or jit):
                batch_size = 1
        data = check_dataset(data)

    # Cấu hình
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), detr=True)
        pad, rect = (0.5, pt)
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
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device, non_blocking=True)
            im = im.half() if half else im.float()
            im /= 255

        # Chuẩn bị nhãn
        bs = len(im)
        batch_idx = targets[:, 0]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]

        _targets = {
            "cls": targets[:, 1].to(device, dtype=torch.long),
            "bboxes": targets[:, 2:].to(device),
            "batch_idx": batch_idx.to(device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
            "mask": masks,
        }

        with dt[1]:
            preds = model(im, batch=_targets, detr=True)
            if compute_loss:
                dec_bboxes, dec_scores, dec_masks, enc_bboxes, enc_scores, enc_masks, dn_meta = preds[1]
                if dn_meta is not None:
                    dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                    dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
                    dn_masks, dec_masks = torch.split(dec_masks, dn_meta["dn_num_split"], dim=2)
                else:
                    dn_bboxes, dn_scores, dn_masks = None, None, None
                
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

        # Lọc topk dự đoán
        bboxes = dec_bboxes[-1]  # [bs, num_queries, 4]
        scores = dec_scores[-1]  # [bs, num_queries, num_classes]
        pred_masks = dec_masks[-1]  # [bs, num_queries, h, w]
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        num_queries = scores.shape[1]
        k = min(max_det, num_queries)  # Đảm bảo k không vượt quá num_queries
        topk_values, topk_indexes = torch.topk(scores.max(dim=-1).values, k, dim=1)
        topk_boxes = topk_indexes
        # Kiểm tra shape trước khi gather
        if topk_boxes.max() >= scores.shape[1]:
            topk_boxes = torch.clamp(topk_boxes, max=scores.shape[1] - 1)
        topk_labels = scores.gather(2, topk_boxes.unsqueeze(-1).repeat(1, 1, scores.shape[-1])).argmax(dim=-1)

        for i in range(bs):
            bbox = bboxes[i, topk_boxes[i]]
            score = topk_values[i]
            cls = topk_labels[i]
            bbox = xywh2xyxy(bbox)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred

        # Đánh giá
        for si, pred in enumerate(outputs):
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

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.tensor(im[si].shape[1:], device=device)[[1, 0, 1, 0]]
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                
                # Đánh giá bbox và mask
                gt_mask = _targets["mask"][targets[:, 0] == si]
                topk_pred_masks = pred_masks[si, topk_boxes[si]]
                correct_bboxes, correct_masks = process_batch(predn, labelsn, iouv, topk_pred_masks, gt_mask, overlap=overlap, masks=True)
                
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            
            stats.append((correct_bboxes, correct_masks, pred[:, 4], pred[:, 5], labels[:, 0]))

            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map, pred_masks[si, topk_boxes[si]].cpu().numpy())

        if plots and batch_i < 3:
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(outputs, max_det=15), pred_masks[:, :15], paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, outputs)

    # Tính toán chỉ số
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
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

    if verbose and nc > 1 and len(stats):
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
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--mask-downsample-ratio', type=int, default=1, help='Downsample ratio for masks')
    parser.add_argument('--overlap', action='store_true', help='Overlap masks in evaluation')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_json |= opt.data.endswith('coco.yaml')
    print_args(vars(opt))
    return opt

def main(opt):
    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))
    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'
        if opt.task == 'speed':
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
        elif opt.task == 'study':
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)
                np.savetxt(f, y, fmt='%10.4g')
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)