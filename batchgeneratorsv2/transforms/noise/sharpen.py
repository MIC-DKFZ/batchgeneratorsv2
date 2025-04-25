import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, List

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class SharpeningTransform(ImageOnlyTransform):
    """
    Applies sharpening to 2D or 3D images using Laplacian-based contrast enhancement.
    Preserves global intensity by explicitly adding scaled Laplacian to the original image.

    Attributes:
        strength (float or Tuple[float, float]): Sharpening strength.
        p_same_for_each_channel (float): Probability of using the same strength for all channels.
        p_per_channel (float): Probability of applying sharpening to a given channel.
    """

    def __init__(self,
                 strength: Union[float, Tuple[float, float]] = 0.2,
                 p_same_for_each_channel: float = 0.0,
                 p_per_channel: float = 1.0,
                 p_clamp_intensities: float = 0):
        super().__init__()
        self.strength = strength
        self.p_same_for_each_channel = p_same_for_each_channel
        self.p_per_channel = p_per_channel
        self.p_clamp_intensities: float = p_clamp_intensities

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        use_same = np.random.rand() < self.p_same_for_each_channel

        if use_same:
            strength = self._sample_strength()
            strengths = [strength] * C
            clamp = np.random.uniform() < self.p_clamp_intensities
            clamp_intensities = [clamp] * C
        else:
            strengths = [self._sample_strength() for _ in range(C)]
            clamp_intensities = [np.random.uniform() < self.p_clamp_intensities for _ in range(C)]

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        return {
            'strengths': strengths,
            'clamp_intensities': clamp_intensities,
            'apply_channel': apply_channel
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        out = img.clone()
        spatial_dims = img.dim() - 1  # 2 for (C, H, W), 3 for (C, D, H, W)

        if spatial_dims == 2:
            kernel = torch.tensor([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=torch.float32, device=img.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
            pad = (1, 1, 1, 1)  # left, right, top, bottom

        elif spatial_dims == 3:
            kernel = torch.tensor([[[0, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 0]],
                                   [[0, -1, 0],
                                    [-1, 6, -1],
                                    [0, -1, 0]],
                                   [[0, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 0]]], dtype=torch.float32, device=img.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3, 3)
            pad = (1, 1, 1, 1, 1, 1)  # left, right, top, bottom, front, back

        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}. Expected 2 or 3.")

        for c, (apply, strength, clamp) in enumerate(zip(params['apply_channel'], params['strengths'], params['clamp_intensities'])):
            if not apply:
                continue

            if clamp:
                mn, mx = torch.min(img[c]), torch.max(img[c])

            x = img[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) or (1, 1, D, H, W)
            padded = F.pad(x, pad, mode='replicate')

            if spatial_dims == 2:
                laplace = F.conv2d(padded, kernel)
            else:
                laplace = F.conv3d(padded, kernel)

            sharpened = x + strength * laplace
            out[c] = sharpened.squeeze()

            if clamp:
                out[c].clamp_(mn, mx)

        return out

    def _sample_strength(self) -> float:
        if isinstance(self.strength, float):
            return self.strength
        return float(np.random.uniform(*self.strength))


if __name__ == '__main__':
    from skimage.data import camera
    from skimage.util import img_as_float32

    # Load camera image and prepare it
    img_np = img_as_float32(camera())  # (H, W), float32, values in [0, 1]
    img_torch = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W) = (C, H, W)

    # Instantiate the transform
    transform = SharpeningTransform(
        strength=(2, 2.1),  # Sharpening strength range
        p_same_for_each_channel=1.0,  # Force same strength for all channels (only 1 channel here)
        p_per_channel=1.0,  # Always apply
        p_clamp_intensities = 1
    )

    # Generate parameters and apply
    params = transform.get_parameters(image=img_torch)
    sharpened = transform._apply_to_image(img_torch, **params)
    from batchviewer import view_batch
    view_batch(img_np, sharpened)