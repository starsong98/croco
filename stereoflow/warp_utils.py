import numpy as np
import torch
import torch.nn as nn


def bwarp(x, flo):
    '''
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    '''
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    #vgrid = torch.autograd.Variable(grid) + flo
    vgrid = grid + flo  # because we wont be backpropagating

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    #mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = torch.ones(x.size()).to(x.device)    # no need for backprop
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    return output * mask


def get_gaussian_weights(x, y, x1, x2, y1, y2, z=1.0):
    # z 0.0 ~ 1.0
    w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(img.device).long().repeat(N, C,1,W).view(-1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(img.device).long().repeat(N, C,H,1).view(-1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(img.device).long().repeat(1, C, H, W).view(-1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(img.device).long().repeat(N, 1, H, W).view(-1)
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(img.device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def fwarp(img, flo):

    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy
        https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
        https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(img, x1, y1, w11)
    img12, o12 = sample_one(img, x1, y2, w12)
    img21, o21 = sample_one(img, x2, y1, w21)
    img22, o22 = sample_one(img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o


def fwarp_wrapper(img, flo):
    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy
        https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
        https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    """
    imgw, o = fwarp(img, flo)   # both are each [N, C, H, W]
    Ft1 = imgw.permute(0, 2, 3, 1)[0].cpu().numpy()    # just to see what this even is
    norm1 = o.permute(0, 2, 3, 1)[0].cpu().numpy()    # just to see what this even is
    holemask = norm1 > 0
    Ft1[norm1 > 0] = Ft1[norm1 > 0] / norm1[norm1 > 0]
    return Ft1, holemask


def bwarp_wrapper(img, flo):
    return