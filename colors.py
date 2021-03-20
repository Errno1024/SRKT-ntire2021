import torch
import torch.cuda
from torch.autograd import Variable
from skimage.color import colorconv

xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])

ycbcr_from_rgb = torch.Tensor([[0.257, -0.148,  0.439],
                               [0.564, -0.291, -0.368],
                               [0.098,  0.439, -0.071]])

rgb_from_ycbcr = ycbcr_from_rgb.inverse()

ycbcr_from_rgb_cons = torch.Tensor([16, 128, 128]) / 255

def rgb2xyz(image: torch.Tensor) -> torch.Tensor:
    while image.ndim < 4:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    mask = image > 0.04045
    r1 = torch.pow((image + 0.055) / 1.055 + 1e-8, 2.4)
    r2 = image / 12.92
    res = mask * r1 + (~mask) * r2
    res = res @ xyz_from_rgb.T.type_as(res).to(res.device)
    image = res.permute(0, 3, 1, 2)
    return image

def get_xyz_coords(illuminant, observer, dtype=torch.float):
    return torch.Tensor(colorconv.get_xyz_coords(illuminant, observer)).type(dtype)

def xyz2lab(image: torch.Tensor, illuminant="D65", observer="2"):
    xyz_ref_white = get_xyz_coords(illuminant, observer, image.dtype)
    image = image.permute(0, 2, 3, 1)
    image = image / xyz_ref_white.to(image.device)
    mask = image > 0.008856
    res = mask * (torch.pow(image + 1e-8, 1 / 3)) + (~mask) * (7.787 * image + 16 / 116)
    x, y, z = res[..., 0], res[..., 1], res[..., 2]
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return torch.stack([L / 100, (a + 128) / 255, (b + 128) / 255], dim=1)

def rgb2lab(image: torch.Tensor, illuminant="D65", observer="2"):
    return xyz2lab(rgb2xyz(image), illuminant, observer)

def rgb2ycbcr(image: torch.Tensor):
    i = image.permute(*range(image.dim() - 3), -2, -1, -3)
    ycbcr = i @ ycbcr_from_rgb + ycbcr_from_rgb_cons
    return ycbcr.permute(*range(ycbcr.dim() - 3), -1, -3, -2)

def ycbcr2rgb(image: torch.Tensor):
    i = image.permute(*range(image.dim() - 3), -2, -1, -3)
    rgb = (i - ycbcr_from_rgb_cons) @ rgb_from_ycbcr
    return rgb.permute(*range(rgb.dim() - 3), -1, -3, -2)

def rgb_gray(image: torch.Tensor):
    i = rgb2ycbcr(image)
    i[..., 1:, :, :] = 0
    return ycbcr2rgb(i)

import numpy as np
import torch.nn.functional as F

def distance(x, y):
    x, y = x.permute(0, 2, 3, 1), y.permute(0, 2, 3, 1)
    r1, g1, b1 = x[..., 0], x[..., 1], x[..., 2]
    r2, g2, b2 = y[..., 0], y[..., 1], y[..., 2]
    ar = (r1 + r2) * 255 / 2
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    return torch.sqrt(1e-8 + (2 + ar / 256) * dr ** 2 + 4 * dg ** 2 + (2 + (255 - ar) / 256) * db ** 2)
