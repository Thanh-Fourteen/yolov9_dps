import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from yolov9.utils.metrics import bbox_iou
from yolov9.utils.general import check_version
from yolov9.utils.loss import FocalLoss


TORCH_1_13 = check_version(torch.__version__, "1.13.0")


def autocast(enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```python
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=True, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 5, "dice": 5}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            device = pred_bboxes.device
            return [(torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Compute the classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Final cost matrix
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]], device=pred_bboxes.device).cumsum_(0)
        return [
            (
                torch.tensor(i, dtype=torch.long, device=pred_bboxes.device), 
                torch.tensor(j, dtype=torch.long, device=pred_bboxes.device) + gt_groups[k]
            )
            for k, (i, j) in enumerate(indices)
        ]
    
    # This function is for future RT-DETR Segment models
    def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
        assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
        # all masks share the same set of points for efficient matching
        sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
        sample_points = 2.0 * sample_points - 1.0
        
        sample_points = sample_points.to(masks.device)
        if masks.dtype != sample_points.dtype:
            sample_points = sample_points.to(masks.dtype)
        out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
        out_mask = out_mask.flatten(0, 1)

        # tgt_mask = torch.cat(gt_mask).unsqueeze(1)
        tgt_mask = torch.unsqueeze(1)
        sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
        sample_points = sample_points.to(tgt_mask.device)
        tgt_mask = tgt_mask.float()
        if tgt_mask.dtype != sample_points.dtype:
            tgt_mask = tgt_mask.to(sample_points.dtype)
        tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    
        with torch.amp.autocast("cuda", enabled=False):
            # binary cross entropy cost
            pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
            neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
            neg_cost_mask = neg_cost_mask.to(tgt_mask.device)
            pos_cost_mask = pos_cost_mask.to(tgt_mask.device)
            cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
            cost_mask /= self.num_sample_points
    
            # dice cost
            out_mask = F.sigmoid(out_mask)
            dv = out_mask.device
            out_mask = out_mask.to(tgt_mask.device)
            numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
            denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)
    
            C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
        return C.to(dv)

    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     device = masks.device  # Lấy device từ masks (cuda:1)

    #     # Tạo sample_points trên cùng device với masks
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2], device=device) * 2.0 - 1.0
    #     if masks.dtype != sample_points.dtype:
    #         sample_points = sample_points.to(dtype=masks.dtype)
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)

    #     # Chuyển gt_mask sang cùng device ngay từ đầu
    #     tgt_mask = torch.cat([m.to(device) for m in gt_mask]).unsqueeze(1).float()
    #     sample_points_gt = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     if tgt_mask.dtype != sample_points_gt.dtype:
    #         sample_points_gt = sample_points_gt.to(dtype=tgt_mask.dtype)
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points_gt, align_corners=False).squeeze([1, 2])

    #     # Tắt mixed precision và tính toán loss
    #     with torch.amp.autocast("cuda", enabled=False):
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         # Không cần .to(tgt_mask.device) vì tất cả đã ở trên cùng device
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, (1 - tgt_mask.T))
    #         cost_mask /= self.num_sample_points

    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0) + 1e-6  # Thêm epsilon
    #         cost_dice = 1 - (numerator + 1) / denominator

    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C  # Không cần .to(dv) vì đã ở đúng device
        

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    targets = targets.to(inputs.device)
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=True, uni_match_ind=0
    ):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 5, "dice": 5}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2, "mask": 5, "dice": 5})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(nn.BCEWithLogitsLoss()) if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """Calculates and returns the bounding box loss and GIoU loss for the predicted and ground truth bounding
        boxes.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss_mask(self, masks, gt_mask, match_indices, gt_groups, postfix=''):
        name_mask = f'loss_mask{postfix}'
        name_dice = f'loss_dice{postfix}'

        b, query, h, w = masks.shape
        loss = {}
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor(0., device=self.device)
            loss[name_dice] = torch.tensor(0., device=self.device)
            return loss
        
        num_gts = sum(gt_groups)
        src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices, gt_groups)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
        target_masks = target_masks.float()  # Chuyển target_masks sang float32
        loss[name_mask] = self.loss_gain['mask'] * sigmoid_focal_loss(src_masks, target_masks, num_gts) / len(target_masks)
        loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts) / len(target_masks)
        return loss

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, gt_groups, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'

    #     b, query, h, w = masks.shape
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     # src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices, gt_groups)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * sigmoid_focal_loss(src_masks, target_masks, num_gts) / len(target_masks)
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts) / len(target_masks)
    #     return loss

    # This function is for future RT-DETR Segment models
    @staticmethod
    def _dice_loss(inputs, targets, num_gts):
        inputs = F.sigmoid(inputs).flatten(1)
        targets = targets.flatten(1)
        targets = targets.to(inputs)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """Get auxiliary losses."""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            loss[3] += loss_[f'loss_mask{postfix}'] 
            loss[4] += loss_[f'loss_dice{postfix}'] 
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, gt_groups, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']
                # print(f"\nloss[3] = {loss[3]}\n")
                # print(f"loss[0] = {loss[0]}\n")

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
            f'loss_mask_aux{postfix}': loss[3],
            f'loss_dice_aux{postfix}': loss[4],
        }
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices, gt_groups=None):
        """
        Assign predicted bounding boxes or masks to ground truth based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes or masks.
            gt_bboxes (torch.Tensor or List[torch.Tensor]): Ground truth bounding boxes or masks.
            match_indices (List[tuple]): List of tuples containing matched indices.
            gt_groups (List[int], optional): List of number of ground truths per image (needed for masks).

        Returns:
            (tuple): Tuple containing assigned predictions and ground truths.
        """
        # Xử lý pred_assigned (dự đoán)
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros((0,) + t.shape[1:], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )

        # Xử lý gt_assigned (ground truth)
        if isinstance(gt_bboxes, torch.Tensor):  # Trường hợp bounding box
            assert gt_groups is not None, "gt_groups must be provided for bounding box assignment"
            gt_groups_cumsum = torch.as_tensor([0, *gt_groups[:-1]], device=self.device).cumsum_(0)
            gt_assigned = torch.cat(
                [
                    (
                        t[(j - gt_groups_cumsum[k]).clamp(max=t.shape[0] - 1)] if len(j) > 0 
                        else torch.zeros((0,) + t.shape[1:], device=self.device)
                    )
                    for t, (_, j), k in zip(gt_bboxes.split(gt_groups, dim=0), match_indices, range(len(gt_groups)))
                ]
            )
        else:  # Trường hợp mask (gt_bboxes là danh sách các tensor)
            assert gt_groups is not None, "gt_groups must be provided for mask assignment"
            gt_groups_cumsum = torch.as_tensor([0, *gt_groups[:-1]], device=self.device).cumsum_(0)
            # Chuyển tất cả tensor trong gt_bboxes sang self.device và kiểm tra
            gt_bboxes = [t.to(self.device, non_blocking=True) for t in gt_bboxes]  # Thêm non_blocking để tối ưu
            gt_assigned = torch.cat(
                [
                    (
                        t[(j - gt_groups_cumsum[k]).clamp(max=t.shape[0] - 1)] if len(j) > 0 
                        else torch.zeros((0,) + t.shape[1:], device=self.device)
                    )
                    for t, (_, j), k in zip(gt_bboxes, match_indices, range(len(gt_bboxes)))
                    if len(t) > 0  # Bỏ qua tensor rỗng
                ]
            ).float()  # Chuyển sang float32 cho mask

        return pred_assigned, gt_assigned

    # def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
    #     """
    #     Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

    #     Args:
    #         pred_bboxes (torch.Tensor): Predicted bounding boxes.
    #         gt_bboxes (torch.Tensor): Ground truth bounding boxes.
    #         match_indices (List[tuple]): List of tuples containing matched indices.

    #     Returns:
    #         (tuple): Tuple containing assigned predictions and ground truths.
    #     """
    #     pred_assigned = torch.cat(
    #         [
    #             t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
    #             for t, (i, _) in zip(pred_bboxes, match_indices)
    #         ]
    #     )
    #     gt_assigned = torch.cat(
    #         [
    #             t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
    #             for t, (_, j) in zip(gt_bboxes, match_indices)
    #         ]
    #     )
    #     return pred_assigned, gt_assigned

    # def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices, gt_groups):
    #     """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""                      
    #     pred_assigned = torch.cat(
    #         [
    #             t[i] if len(i) > 0 else torch.zeros((0,) + t.shape[1:], device=self.device)
    #             for t, (i, _) in zip(pred_bboxes, match_indices)
    #         ]
    #     )

    #     gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
    #     gt_assigned = torch.cat(
    #         [
    #             (t[j - gt_groups[k]] if len(j) > 0 else torch.zeros(((0,) + t.shape[1:]), device=t.device))
    #             for t, (_, j), k in zip(gt_bboxes, match_indices, range(len(gt_bboxes)))
    #         ]
    #     )
    #     return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """Get losses."""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(self._get_loss_mask(masks, gt_mask, match_indices, gt_groups, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, pred_masks, batch, postfix="", **kwargs):
        """
        Args:
            pred_bboxes (torch.Tensor): [l, b, query, 4]
            pred_scores (torch.Tensor): [l, b, query, num_classes]
            batch (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                gt_groups (List(int)): a list of batch size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls = batch["cls"].to(self.device)
        gt_bboxes = batch["bboxes"].to(self.device)
        gt_groups = batch["gt_groups"]  # Danh sách int, không cần to(device)
        gt_masks = batch["mask"].to(self.device, non_blocking=True)

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, 
            masks=pred_masks[-1], gt_mask=gt_masks, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, 
                    match_indices, postfix, masks=pred_masks[:-1], gt_mask=gt_masks
                )
            )

        return total_loss

class RTDETRSegmentLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_masks = None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores, pred_masks = preds
        total_loss = super().forward(pred_bboxes, pred_scores, pred_masks, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, dn_masks, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices