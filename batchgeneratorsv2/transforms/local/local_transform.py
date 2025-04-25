import numpy as np
import scipy.stats as st
from abc import ABC
from typing import Tuple
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar


class LocalTransform(ABC):
    def __init__(self, scale: RandomScalar, loc: RandomScalar = (-1, 2)):
        self.loc = loc
        self.scale = scale

    def _generate_kernel(self, img_shp: Tuple[int, ...]) -> np.ndarray:
        ndim = len(img_shp)
        x_grids = [np.arange(-0.5, s + 0.5, dtype=np.float32) for s in img_shp]
        kernels = []

        for d in range(ndim):
            loc_val = sample_scalar(self.loc, img_shp, d)
            scale_val = sample_scalar(self.scale, img_shp, d)
            loc_rescaled = loc_val * img_shp[d]
            cdf = st.norm.cdf(x_grids[d], loc=loc_rescaled, scale=scale_val)
            kernels.append(np.diff(cdf).astype(np.float32))

        kernel = kernels[0][:, None] @ kernels[1][None]
        if ndim == 3:
            kernel = kernel[:, :, None] @ kernels[2][None]

        kernel -= kernel.min()
        kernel_max = kernel.max()
        if kernel_max > 0:
            kernel /= kernel_max
        return kernel

    def _generate_multiple_kernel_image(self, img_shp: Tuple[int, ...], num_kernels: int) -> np.ndarray:
        """
        Places multiple additive Gaussians in the image and normalizes the sum to [0, 1].

        Parameters:
            img_shp (Tuple[int, ...]): Spatial shape (e.g., (X, Y[, Z]))
            num_kernels (int): Number of kernels to generate and sum

        Returns:
            np.ndarray: Combined kernel image with values in [0, 1]
        """
        kernel_image = np.zeros(img_shp, dtype=np.float32)
        for _ in range(num_kernels):
            kernel_image += self._generate_kernel(img_shp)

        kernel_image -= kernel_image.min()
        kernel_max = kernel_image.max()
        if kernel_max > 0:
            kernel_image /= kernel_max
        return kernel_image

    @staticmethod
    def invert_kernel(kernel_image: np.ndarray) -> np.ndarray:
        """
        Inverts a normalized kernel: 1 - kernel

        Assumes input is in [0, 1].

        Parameters:
            kernel_image (np.ndarray): Input kernel in [0, 1]

        Returns:
            np.ndarray: Inverted kernel in [0, 1]
        """
        return 1.0 - kernel_image

    @staticmethod
    def run_interpolation(original_image: np.ndarray,
                          modified_image: np.ndarray,
                          kernel_image: np.ndarray) -> np.ndarray:
        """
        Blends original and modified images using the given kernel as a per-pixel weight map.

        Parameters:
            original_image (np.ndarray): Unmodified input image
            modified_image (np.ndarray): Modified version (e.g., gamma-corrected)
            kernel_image (np.ndarray): Kernel in [0, 1], where 0 = keep original, 1 = keep modified

        Returns:
            np.ndarray: Blended result
        """
        return original_image * (1.0 - kernel_image) + modified_image * kernel_image