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