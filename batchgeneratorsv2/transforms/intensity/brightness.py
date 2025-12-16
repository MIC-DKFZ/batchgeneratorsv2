import numpy as np
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class MultiplicativeBrightnessTransform(ImageOnlyTransform):
    def __init__(self, multiplier_range: RandomScalar, synchronize_channels: bool, p_per_channel: float = 1):
        super().__init__()
        self.multiplier_range = multiplier_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        if self.synchronize_channels:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data_dict['image'], channel=None)] * len(apply_to_channel))
        else:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data_dict['image'], channel=c) for c in apply_to_channel])
        return {
            'apply_to_channel': apply_to_channel,
            'multipliers': multipliers
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params['apply_to_channel']) == 0:
            return img
        # even though this is array notation it's a lot slower. Shame shame
        # img[params['apply_to_channel']] *= params['multipliers'].view(-1, *[1]*(img.ndim - 1))
        for c, m in zip(params['apply_to_channel'], params['multipliers']):
            img[c] *= m
        return img


class BrightnessAdditiveTransform(ImageOnlyTransform):
    """
    Adds random additive brightness noise sampled from a Gaussian distribution (mu, sigma).

    Supports either synchronized brightness shift across all channels or per-channel brightness shift.

    Args:
        mu (float): Mean of the Gaussian used to sample brightness shifts.
        sigma (float): Standard deviation of the Gaussian.
        synchronize_channels (bool): If True, brightness shifts are shared across all channels.
        p_per_channel (float): Probability to apply the brightness shift to each channel.
    """

    def __init__(self,
                 mu: float,
                 sigma: float,
                 synchronize_channels: bool = True,  # Changed to synchronize_channels
                 p_per_channel: float = 1.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.synchronize_channels = synchronize_channels  # Now it's being used
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        c = img.shape[0]
        apply_to_channel = (torch.rand(c, device=img.device) < self.p_per_channel).nonzero(as_tuple=False).flatten()

        if len(apply_to_channel) == 0:
            return {"apply_to_channel": apply_to_channel, "shift": None}

        # Apply either synchronized or per-channel brightness shift
        if self.synchronize_channels:
            shift_value = float(sample_scalar((self.mu, self.sigma), image=img, channel=None))
            shift = torch.full((c,), shift_value, device=img.device)
        else:
            shift = torch.stack([torch.normal(self.mu, self.sigma) for _ in range(c)], dim=0).to(img.device)

        return {"apply_to_channel": apply_to_channel, "shift": shift}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params["shift"] is None:
            return img

        apply_idx = params["apply_to_channel"]
        if apply_idx.numel() == 0:
            return img

        shift = params["shift"]
        # Build full per-channel shift vector; non-selected channels get shift 0
        shift_full = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)
        shift_full[apply_idx] = shift[apply_idx]

        view_shape = (img.shape[0],) + (1,) * (img.ndim - 1)
        img.add_(shift_full.view(view_shape))
        return img


if __name__ == '__main__':
    from time import time
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # mbt = BrightnessAdditiveTransform(0, 0.5,True, 1)
    mbt = MultiplicativeBrightnessTransform((0.5, 2),False, 1)

    times_torch = []
    for _ in range(1000):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform

    gnt_bg = BrightnessMultiplicativeTransform((0.5, 2), True, p_per_sample=1)
    times_bg = []
    for _ in range(1000):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))
