import numpy as np
from functools import lru_cache

import torch
from torch.nn.functional import pad, conv3d, conv1d, conv2d

from batchgenerators_torch.helpers.scalar_type import ScalarType, sample_scalar
from batchgenerators_torch.transforms.base.basic_transform import ImageOnlyTransform


class GaussianBlurTransform(ImageOnlyTransform):
    def __init__(self,
                 blur_sigma: ScalarType = (1, 5),
                 synchronize_channels: bool = False,
                 synchronize_axes: bool = False,
                 p_per_channel: float = 1
                 ):
        """
        uses separable gaussian filters for all the speed

        blur_sigma, if callable, will be called as blur_sigma(image, shape, dim) where shape is (c, x(, y, z) and dim i
        s 1, 2 or 3 for x, y and z, respectively)
        :param blur_sigma:
        :param synchronize_channels:
        :param synchronize_axes:
        :param p_per_channel:
        """
        super().__init__()
        self.blur_sigma = blur_sigma
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        dims = len(shape) - 1
        dct = {}
        dct['apply_to_channel'] = torch.rand(shape[0]) < self.p_per_channel
        if self.synchronize_axes:
            dct['kernel_sizes'] = \
                [[sample_scalar(self.blur_sigma, shape, dim=None)] * dims
                 for _ in range(sum(dct['apply_to_channel']))] \
                    if not self.synchronize_channels else \
                    [sample_scalar(self.blur_sigma, shape, dim=None)] * dims
        else:
            dct['kernel_sizes'] = \
                [[sample_scalar(self.blur_sigma, shape, dim=i + 1) for i in range(dims)]
                 for _ in range(sum(dct['apply_to_channel']))] \
                    if not self.synchronize_channels else \
                    [sample_scalar(self.blur_sigma, shape[i + 1]) for i in range(dims)]
        return dct

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        dim = len(img.shape[1:])
        print(params['kernel_sizes'])
        if self.synchronize_channels:
            # we can compute that in one go as the conv implementation supports arbitrary input channels (with expanded kernel)
            for d in range(dim):
                print(d, params['kernel_sizes'][d])
                img[params['apply_to_channel']] = self.blur_dimension(img[params['apply_to_channel']], params['kernel_sizes'][d], d)
        else:
            # we have to go through all the channels, build the kernel for each channel etc
            idx = np.where(params['apply_to_channel'])[0]
            for i in idx:
                for d in range(dim):
                    print(i, d, params['kernel_sizes'][i][d])
                    img[i:i+1] = self.blur_dimension(img[i:i+1], params['kernel_sizes'][i][d], d)
        return img

    def blur_dimension(self, img: torch.Tensor, sigma: float, dim_to_blur: int):
        """
        Smoothes an input image with a 1D Gaussian kernel along the specified dimension.
        The function supports 1D, 2D, and 3D images.

        :param img: Input image tensor with shape (C, X), (C, X, Y), or (C, X, Y, Z),
                    where C is the channel dimension and X, Y, Z are spatial dimensions.
        :param sigma: The standard deviation of the Gaussian kernel.
        :param dim_to_blur: The dimension along which to apply the Gaussian blur (0 for X, 1 for Y, 2 for Z).
        :return: The blurred image tensor.
        """
        assert img.ndim - 1 > dim_to_blur, "dim_to_blur must be a valid spatial dimension of the input image."

        # Adjustments for kernel based on image dimensions
        spatial_dims = img.ndim - 1  # Number of spatial dimensions in the input image
        kernel = _build_kernel(sigma)
        ksize = kernel.shape[0]

        # Dynamically set up padding, convolution operation, and kernel shape based on the number of spatial dimensions
        conv_ops = {1: conv1d, 2: conv2d, 3: conv3d}
        conv_op = conv_ops[spatial_dims]

        # Adjust kernel and padding for the specified blur dimension and input dimensions
        if spatial_dims == 1:
            kernel = kernel[None, None, :]
            padding = [ksize // 2, ksize // 2]
        elif spatial_dims == 2:
            if dim_to_blur == 0:
                kernel = kernel[None, None, :, None]
                padding = [0, 0, ksize // 2, ksize // 2]
            else:  # dim_to_blur == 1
                kernel = kernel[None, None, None, :]
                padding = [ksize // 2, ksize // 2, 0, 0]
        else:  # spatial_dims == 3
            # Expand kernel and adjust padding based on the blur dimension
            if dim_to_blur == 0:
                kernel = kernel[None, None, :, None, None]
                padding = [0, 0, 0, 0, ksize // 2, ksize // 2]
            elif dim_to_blur == 1:
                kernel = kernel[None, None, None, :, None]
                padding = [0, 0, ksize // 2, ksize // 2, 0, 0]
            else:  # dim_to_blur == 2
                kernel = kernel[None, None, None, None, :]
                padding = [ksize // 2, ksize // 2, 0, 0, 0, 0]

        # Apply padding
        img_padded = pad(img, padding, mode="reflect")

        # Apply convolution
        # remember that weights are [c_out, c_in, ...]
        img_blurred = conv_op(img_padded[None], kernel.expand(img_padded.shape[0], *[-1] * (kernel.ndim - 1)), groups=img_padded.shape[0])[0]

        return img_blurred


# kernels are small, we can justify remembering the last 50 or so
@lru_cache(maxsize=50)
def _build_kernel(sigma: float) -> torch.Tensor:
    kernel_size = _compute_kernel_size(sigma)
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _round_to_nearest_odd(n):
    rounded = round(n)
    # If the rounded number is odd, return it
    if rounded % 2 == 1:
        return rounded
    # If the rounded number is even, adjust to the nearest odd number
    return rounded + 1 if n - rounded >= 0 else rounded - 1


def _compute_kernel_size(sigma, truncate: float = 4):
    ksize = _round_to_nearest_odd(sigma * truncate + 0.5)
    return ksize


if __name__ == "__main__":
    gnt = GaussianBlurTransform((1, 20), False, True, 1)

    data_dict = {'image': torch.ones((2, 64, 64, 64))}
    data_dict['image'][:, :32, :32, :32] = 200
    out = gnt(**data_dict)
