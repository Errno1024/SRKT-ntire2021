from torch import nn
import torch.nn.functional
import torch.nn.functional as F
import numpy as np
import torch.fft

from pytorch_msssim import SSIM
from pytorch_msssim import MS_SSIM as MSSSIM

def l1loss(x, y):
    return torch.nn.functional.l1_loss(x, y, reduction='mean')

def l2loss(x, y):
    return torch.pow(x - y, 2).mean()

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (10 * torch.log10(x.shape[-2] * x.shape[-1] / (x - y).pow(2).sum(dim=(2, 3)))).mean(dim=1)

ssim = SSIM(data_range=1.0)
msssim = MSSSIM(data_range=1.0)

def gaussian(x, sigma=1.0):
    return np.exp(-(x**2) / (2*(sigma**2)))

def build_gauss_kernel(
        size=5, sigma=1.0, n_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur
    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    Also, in_channels == out_channels == n_channels
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.mgrid[range(size), range(size)] - size//2
    kernel = np.prod(gaussian(grid, sigma), axis=0)
    # kernel = np.sum(gaussian(grid, sigma), axis=0)
    kernel /= np.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = np.tile(kernel, (n_channels, 1, 1, 1))
    kernel = torch.from_numpy(kernel).to(torch.float).to(device)
    return kernel

def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    _, n_channels, _, _ = images.shape
    _, _, kw, kh = kernel.shape
    imgs_padded = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(imgs_padded, kernel, groups=n_channels)

def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image
    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid

class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                n_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss

import colors

def labloss(x, y):
    return l1loss(colors.rgb2lab(x), colors.rgb2lab(y))

def lab2loss(x, y):
    return l2loss(colors.rgb2lab(x), colors.rgb2lab(y))

laploss = LapLoss()

def color_distance_loss(x, y, l=1):
    return (colors.distance(x, y) ** l).mean()
