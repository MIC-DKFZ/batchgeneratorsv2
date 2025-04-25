import torch
import numpy as np
from typing import List

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.local.local_transform import LocalTransform


class BrightnessGradientAdditiveTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies a localized brightness modulation to an image using a smooth Gaussian gradient.

    This transform creates a spatial Gaussian kernel (in 2D or 3D), optionally zero-centers it,
    scales its peak intensity, and adds it to the image. This can simulate intensity drift,
    local contrast changes, or smooth lighting artifacts.

    The effect is applied per channel, and each channel can have a different gradient or share the same one.

    ---
    Example use cases:
    - Simulating local contrast shifts in MRI
    - Adding spatial brightness gradients for robustness
    - Mimicking smooth scanner inhomogeneity fields

    Args:
        scale (RandomScalar):
            Controls the spatial spread of the Gaussian kernel (standard deviation).
            Can be:
              - float: fixed spread
              - (min, max): uniformly sampled per-dimension
              - callable(image_shape, dim): custom sampling per axis

        loc (RandomScalar):
            Controls the relative location of the Gaussian kernel (in percentage of image size).
            Can be:
              - (min, max): e.g. (-1, 2) allows centers to be far outside the image for smoother edges
              - callable(image_shape, dim): custom sampling per axis

        max_strength (RandomScalar):
            Peak value of the additive brightness change (positive or negative depending on the Gaussian).
            Can be:
              - float: fixed strength
              - (min, max): sampled strength
              - callable(image, kernel): fully custom

        same_for_all_channels (bool):
            If True, one shared kernel is used across all channels.
            If False, each channel gets its own random kernel and strength.

        mean_centered (bool):
            If True, the Gaussian kernel is mean-centered (i.e., ∑kernel = 0),
            which ensures the overall mean intensity of the image stays constant.

        clip_intensities (bool):
            If True, clamps image values after modification to their original min/max.
            Useful to prevent range overflow.

        p_per_channel (float):
            Probability to apply the transform to each channel independently.

    Returns:
        Modified image of the same shape with localized brightness modulation applied.

    Example:
        transform = BrightnessGradientAdditiveTransform(
            scale=(5, 15),
            max_strength=(0.1, 0.5),
            same_for_all_channels=True,
            mean_centered=True
        )
    """
    def __init__(self,
                 scale: RandomScalar,
                 loc: RandomScalar = (-1, 2),
                 max_strength: RandomScalar = 1.0,
                 same_for_all_channels: bool = True,
                 mean_centered: bool = True,
                 clip_intensities: bool = False,
                 p_per_channel: float = 1.0):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.max_strength = max_strength
        self.same_for_all_channels = same_for_all_channels
        self.mean_centered = mean_centered
        self.clip_intensities = clip_intensities
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        # Early exit if nothing will be applied
        if not any(apply_channel):
            return {'kernels': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(spatial)
            if self.mean_centered:
                kernel -= kernel.mean()

            max_abs = np.abs(kernel).max()
            if max_abs < 1e-8:
                return {'kernels': [None] * C}

            strength = sample_scalar(self.max_strength, image, kernel)
            if strength == 0.0:
                return {'kernels': [None] * C}

            kernel /= max_abs
            kernel *= strength

            kernels = [kernel if apply else None for apply in apply_channel]

        else:
            kernels = []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    continue

                kernel = self._generate_kernel(spatial)
                if self.mean_centered:
                    kernel -= kernel.mean()
                max_abs = np.abs(kernel).max()
                if max_abs < 1e-8:
                    kernels.append(None)
                    continue

                strength = sample_scalar(self.max_strength, image, kernel)
                if strength == 0.0:
                    kernels.append(None)
                    continue

                kernel /= max_abs
                kernel *= strength
                kernels.append(kernel)

        return {'kernels': kernels}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c, kernel in enumerate(params['kernels']):
            if kernel is None:
                continue
            kernel_tensor = torch.from_numpy(kernel).to(img.device, dtype=img.dtype)
            img[c].add_(kernel_tensor)

        if self.clip_intensities:
            img.clamp_(min=img.min(), max=img.max())

        return img

if __name__ == '__main__':
    import torch
    from batchviewer import view_batch

    # Create synthetic z-score normalized 3D image (C, D, H, W)
    image = torch.randn(1, 32, 64, 64)  # single-channel 3D volume

    # Instantiate the transform
    transform = BrightnessGradientAdditiveTransform(
        scale=(25, 50),  # controls width of Gaussian
        loc=(-0.5, 1.5),
        max_strength=(2, 5),  # how strong the modulation is
        same_for_all_channels=True,
        mean_centered=True,
        clip_intensities=False,
        p_per_channel=1.0  # always apply
    )

    # Get transform parameters and apply
    params = transform.get_parameters(image=image)
    image_modulated = transform._apply_to_image(image.clone(), **params)

    # Visualize with your preferred viewer
    view_batch(image, image_modulated)