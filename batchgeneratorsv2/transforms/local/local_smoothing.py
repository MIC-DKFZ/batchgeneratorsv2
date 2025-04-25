import torch
import numpy as np
from batchgeneratorsv2.transforms.local.local_transform import LocalTransform
from scipy.ndimage import gaussian_filter
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar


class LocalSmoothingTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies localized Gaussian smoothing to parts of the image using a spatial Gaussian mask.

    A blurred copy of the image is interpolated with the original, weighted by a Gaussian kernel.
    The strength and extent of the blur are both controllable.

    Args:
        scale (RandomScalar): Gaussian spread for the spatial weighting mask.
        loc (RandomScalar): Relative center position for the Gaussian (in % of image size).
        smoothing_strength (RandomScalar): Max weight of the smoothed image in the interpolation [0, 1].
        kernel_size (RandomScalar): Sigma for the actual Gaussian smoothing of the image.
        same_for_all_channels (bool): Whether to apply the same kernel to all channels.
        p_per_channel (float): Probability of applying transform per channel.
    """

    def __init__(self,
                 scale: RandomScalar,
                 loc: RandomScalar = (-1, 2),
                 smoothing_strength: RandomScalar = (0.5, 1.0),
                 kernel_size: RandomScalar = (0.5, 1.5),
                 same_for_all_channels: bool = True,
                 p_per_channel: float = 1.0):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.smoothing_strength = smoothing_strength
        self.kernel_size = kernel_size
        self.same_for_all_channels = same_for_all_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'sigma': None, 'strengths': [None] * C}

        sigma = sample_scalar(self.kernel_size)

        if self.same_for_all_channels:
            kernel = self._generate_kernel(spatial).astype(np.float32)
            strength = sample_scalar(self.smoothing_strength)

            kernels = [kernel if apply else None for apply in apply_channel]
            strengths = [strength if apply else None for apply in apply_channel]
        else:
            kernels, strengths = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    strengths.append(None)
                    continue
                kernel = self._generate_kernel(spatial).astype(np.float32)
                strength = sample_scalar(self.smoothing_strength)
                kernels.append(kernel)
                strengths.append(strength)

        return {'kernels': kernels, 'sigma': sigma, 'strengths': strengths}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        img_np = img.cpu().numpy()
        sigma = params['sigma']

        for c, (kernel, strength) in enumerate(zip(params['kernels'], params['strengths'])):
            if kernel is None:
                continue

            kernel = kernel * strength  # scale kernel by smoothing strength
            smoothed = gaussian_filter(img_np[c], sigma=sigma)
            img_np[c] = self.run_interpolation(img_np[c], smoothed, kernel)

        return torch.from_numpy(img_np).to(img.device, dtype=img.dtype)


if __name__ == '__main__':
    from batchviewer import view_batch

    # Single-channel synthetic volume
    image = torch.rand(1, 32, 64, 64)

    # Or contrast
    smoother = LocalSmoothingTransform(loc=(0, 1), scale=(10, 20), kernel_size=(3, 10), p_per_channel=1.0)

    # Apply either one
    params = smoother.get_parameters(image=image)
    image_aug = smoother._apply_to_image(image.clone(), **params)

    view_batch(image, image_aug)

