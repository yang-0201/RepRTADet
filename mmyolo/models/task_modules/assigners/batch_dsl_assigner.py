# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from mmyolo.models.losses import bbox_overlaps
INF = 100000000
EPS = 1.0e-7
class BboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str



    def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        """Calculate overlap between two set of bboxes.

        If ``is_aligned `` is ``False``, then calculate the overlaps between each
        bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
        pair of bboxes1 and bboxes2.

        Args:
            bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
            bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection over
                foreground) or "giou" (generalized intersection over union).
                Default "iou".
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Default 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

        Example:
            >>> bboxes1 = torch.FloatTensor([
            >>>     [0, 0, 10, 10],
            >>>     [10, 10, 20, 20],
            >>>     [32, 32, 38, 42],
            >>> ])
            >>> bboxes2 = torch.FloatTensor([
            >>>     [0, 0, 10, 20],
            >>>     [0, 10, 10, 19],
            >>>     [10, 10, 20, 20],
            >>> ])
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
            >>> assert overlaps.shape == (3, 3)
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
            >>> assert overlaps.shape == (3, )

        Example:
            >>> empty = torch.empty(0, 4)
            >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
            >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
            >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
            >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
        """

        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        # Either the boxes are empty or the length of boxes's last dimenstion is 4
        assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            if is_aligned:
                return bboxes1.new(batch_shape + (rows, ))
            else:
                return bboxes1.new(batch_shape + (rows, cols))

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

        if is_aligned:
            lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
            rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

            wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1 + area2 - overlap
            else:
                union = area1
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        else:
            lt = torch.max(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
            rb = torch.min(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

            wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1[..., None] + area2[..., None, :] - overlap
            else:
                union = area1[..., None]
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                        bboxes2[..., None, :, :2])
                enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                        bboxes2[..., None, :, 2:])

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        if mode in ['iou', 'iof']:
            return ious
        # calculate gious
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area
        return gious
def find_inside_points(boxes: Tensor,
                       points: Tensor,
                       box_dim: int = 4,
                       eps: float = 0.01) -> Tensor:
    """Find inside box points in batches. Boxes dimension must be 3.

    Args:
        boxes (Tensor): Boxes tensor. Must be batch input.
            Has shape of (batch_size, n_boxes, box_dim).
        points (Tensor): Points coordinates. Has shape of (n_points, 2).
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.
        eps (float): Make sure the points are inside not on the boundary.
            Only use in rotated boxes. Defaults to 0.01.

    Returns:
        Tensor: A BoolTensor indicating whether a point is inside
        boxes. The index has shape of (n_points, batch_size, n_boxes).
    """
    if box_dim == 4:
        # Horizontal Boxes
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0

    elif box_dim == 5:
        # Rotated Boxes
        points = points[:, None, None]
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        is_in_gts = (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
                    (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)
    else:
        raise NotImplementedError(f'Unsupport box_dim:{box_dim}')

    return is_in_gts
def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    

    center1 = pred[..., :2]
    center2 = target[..., :2]

    whs = center1[..., :2] - center2[..., :2]

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #
    print(pred.size())
    w1 = pred[..., 2]  + eps
    h1 = pred[..., 3]  + eps
    w2 = target[..., 2] + eps
    h2 = target[..., 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


def get_box_center(boxes: Tensor, box_dim: int = 4) -> Tensor:
    """Return a tensor representing the centers of boxes.

    Args:
        boxes (Tensor): Boxes tensor. Has shape of (b, n, box_dim)
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.

    Returns:
        Tensor: Centers have shape of (b, n, 2)
    """
    if box_dim == 4:
        # Horizontal Boxes, (x1, y1, x2, y2)
        return (boxes[..., :2] + boxes[..., 2:]) / 2.0
    elif box_dim == 5:
        # Rotated Boxes, (x, y, w, h, a)
        return boxes[..., :2]
    else:
        raise NotImplementedError(f'Unsupported box_dim:{box_dim}')


@TASK_UTILS.register_module()
class BatchDynamicSoftLabelAssigner(nn.Module):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        num_classes (int): number of class
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
        batch_iou (bool): Use batch input when calculate IoU.
            If set to False use loop instead. Defaults to True.
    """

    def __init__(
        self,
        num_classes,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D'),
        batch_iou: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        # self.iou_calculator = BboxOverlaps2D
        self.batch_iou = batch_iou
    


    @torch.no_grad()
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor,
                gt_labels: Tensor, gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()

        if num_gt == 0 or num_bboxes == 0:
            return {
                'assigned_labels':
                gt_labels.new_full(
                    pred_scores[..., 0].shape,
                    self.num_classes,
                    dtype=torch.long),
                'assigned_labels_weights':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 1),
                'assigned_bboxes':
                gt_bboxes.new_full(pred_bboxes.shape, 0),
                'assign_metrics':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
            }

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(
                f'type of {type(gt_bboxes)} are not implemented !')
        else:
            is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)

        # (N_points, B, N_boxes)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        # (N_points, B, N_boxes) -> (B, N_points, N_boxes)
        is_in_gts = is_in_gts.permute(1, 0, 2)
        # (B, N_points)
        valid_mask = is_in_gts.sum(dim=-1) > 0

        gt_center = get_box_center(gt_bboxes, box_dim)

        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] -
                    gt_center[:, None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[None, :, None]

        # prevent overflow
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        if self.batch_iou:

            pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)

        else:
            ious = []
            for box, gt in zip(decoded_bboxes, gt_bboxes):
                iou = bbox_overlaps(
                pred = box,
                target = gt,
                iou_mode='ciou',
                bbox_format='xyxy').clamp(0)
                # iou = self.iou_calculator(box, gt)
                ious.append(iou)
            pairwise_ious = torch.stack(ious, dim=0)

        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        # select the predicted scores corresponded to the gt_labels
        pairwise_pred_scores = pred_scores.permute(0, 2, 1)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0],
                                                    idx[1]].permute(0, 2, 1)
        # classification cost
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious,
            reduction='none') * scale_factor.abs().pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior

        max_pad_value = torch.ones_like(cost_matrix) * INF
        cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt),
                                  cost_matrix, max_pad_value)

        (matched_pred_ious, matched_gt_inds,
         fg_mask_inboxes) = self.dynamic_k_matching(cost_matrix, pairwise_ious,
                                                    pad_bbox_flag)

        del pairwise_ious, cost_matrix

        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]

        assigned_labels = gt_labels.new_full(pred_scores[..., 0].shape,
                                             self.num_classes)
        assigned_labels[fg_mask_inboxes] = gt_labels[
            batch_index, matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()

        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape,
                                                     1)

        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index,
                                                     matched_gt_inds]

        assign_metrics = gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious

        return dict(
            assigned_labels=assigned_labels,
            assigned_labels_weights=assigned_labels_weights,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics)

    # @torch.no_grad()
    

    def dynamic_k_matching(
            self, cost_matrix: Tensor, pairwise_ious: Tensor,
            pad_bbox_flag: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        num_gts = pad_bbox_flag.sum((1, 2)).int()
        # sorting the batch cost matirx is faster than topk
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for b in range(pad_bbox_flag.shape[0]):
            for gt_idx in range(num_gts[b]):
                topk_ids = sorted_indices[b, :dynamic_ks[b, gt_idx], gt_idx]
                matching_matrix[b, :, gt_idx][topk_ids] = 1

        del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(2)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes
