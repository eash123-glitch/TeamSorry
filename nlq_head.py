import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.module):

    def __init__(
            self,
            num_channels,
            eps = 1e-5,
            affine = True,
            device = None,
            dtype = None,
    ):
        
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() ==3
        assert x.shape[1] == self.num_channels

        # norm over the channel dimension(C)

        mu = torch.mean(x, dim =1, keepdim = True)
        sigma = torch.mean((x - mu)**2, dim =1, keepdim = True)
        out = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affine:
            out = out * self.weight + self.bias
        return out
    

    def ctr_diou_loss_1d(
            input_offsets : torch.Tensor,
            target_offsets : torch.Tensor,
            reduction : str = 'none',
            eps : float = 1e-8
    ) -> torch.Tensor:
        """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
        
        
        input_offsets = input_offsets.float()
        target_offsets = target_offsets.float()
        assert(input_offsets >= 0.0).all(), "Input offsets must be non-negative"
        assert(target_offsets >= 0.0).all(), "Target offsets must be non-negative"

        lp, rp = input_offsets[:, 0], input_offsets[:, 1]
        lt, rt = target_offsets[:, 0], target_offsets[:, 1]

        lkis = torch.min(lp, lt)
        rkis = torch.min(rp, rt)

        inter = lkis + rkis
        union = (lp + rp) + (lt + rt) - inter
        iou = inter /  union.clamp(min = eps)

        lc = torch.max(lp, lt)
        rc = torch.max(rp, rt)
        length_c = lc + rc

        rho = 0.5 * (rp - lp - rt + lt)
        loss = 1.0 - iou + torch.square(rho / length_c.clamp(min = eps))

        if reduction == 'mean':
            loss = loss.mean() if loss.numel() > 0 else 0.0*loss.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss
    
    def sigmoid_focal_loss(
            inputs: torch.Tensor,
            targets: torch.Tensor,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = "none",
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
        
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        
        elif reduction == "sum":    
            loss = loss.sum()
        
        return loss
    

class BufferList(nn.Module):

    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):

        super().__init__()
        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent = False)

    def __len__(self):
        return len(self._buffers)
    
    def __iter__(self):
        return iter(self._buffers.values())
    

class PointGenerator(nn.Module):

    """
    Generate points according to feature map sizes
    """

    def __init__(
            self,
            max_seq_len,
            fpn_strides,
            regression_range,
            use_offset = False
    ):
        super().__init__()

        fpn_levels = len(fpn_strides)
        assert len(regression_range) == fpn_levels

        self.max_seq_len = max_seq_len
        self.fpn_strides = fpn_strides
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.fpn_levels = fpn_levels
        self.buffer_points = self.generate_points()

    def generate_points(self):
        
        points_list =[]
        for l, stride in enumerate(self.fpn_strides):
            reg_range = torch.as_tensor(self.regression_range[l], dtype = torch.float32)
            fpn_stride = torch.as_tensor(stride, dtype = torch.float32)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]

            if self.use_offset:
                points += stride // 2

            req_range = req_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride.repeat(points.shape[0], 1)
            points_list.append(torch.cat([points, req_range, fpn_stride], dim =1))

        return BufferList(points_list)
    
    def forward(self, features):

        assert len(features) == self.fpn_levels
        points_list = []
        feat_lens = [f.shape[-1] for f in features]
        for feat_len, buffer_points in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_points.shape[0]
            points = buffer_points[:feat_len, :]
            points_list.append(points)
        return points_list