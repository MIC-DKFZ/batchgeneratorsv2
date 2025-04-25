import torch
import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.local.local_transform import LocalTransform


class LocalContrastTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies localized contrast modification using a spatial Gaussian mask.

    A contrast-modified version of the image is blended with the original using a kernel-based interpolation.

    Args:
        scale (RandomScalar): Gaussian spread for the spatial weighting mask.
        loc (RandomScalar): Relative center position for the Gaussian (in % of image size).
        new_contrast (RandomScalar): Multiplicative factor for local contrast. 1.0 = no change.
        same_for_all_channels (bool): Whether to use one kernel/contrast value for all channels.
        p_per_channel (float): Probability to apply to each channel.
    """

    def __init__(self,
                 scale: RandomScalar,
                 loc: RandomScalar = (-1, 2),
                 new_contrast: RandomScalar = (0.5, 1.5),
                 same_for_all_channels: bool = True,
                 p_per_channel: float = 1.0):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.new_contrast = new_contrast
        self.same_for_all_channels = same_for_all_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'contrasts': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(spatial).astype(np.float32)
            contrast = sample_scalar(self.new_contrast)

            kernels = [kernel if apply else None for apply in apply_channel]
            contrasts = [contrast if apply else None for apply in apply_channel]
        else:
            kernels, contrasts = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    contrasts.append(None)
                    continue
                kernel = self._generate_kernel(spatial).astype(np.float32)
                contrast = sample_scalar(self.new_contrast)
                kernels.append(kernel)
                contrasts.append(contrast)

        return {'kernels': kernels, 'contrasts': contrasts}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        img_np = img.cpu().numpy()

        for c, (kernel, contrast) in enumerate(zip(params['kernels'], params['contrasts'])):
            if kernel is None:
                continue

            channel = img_np[c]
            mean = (channel * kernel).sum() / (kernel.sum() + 1e-8)
            modified = (channel - mean) * contrast + mean
            img_np[c] = self.run_interpolation(channel, modified, kernel)

        return torch.from_numpy(img_np).to(img.device, dtype=img.dtype)


if __name__ == '__main__':
    from batchviewer import view_batch

    # Single-channel synthetic volume
    image = torch.rand(1, 32, 64, 64)

    # Or contrast
    contrast = LocalContrastTransform(scale=(10, 20), new_contrast=(0.3, 2.0), p_per_channel=1.0)

    # Apply either one
    params = contrast.get_parameters(image=image)
    image_aug = contrast._apply_to_image(image.clone(), **params)

    view_batch(image, image_aug)