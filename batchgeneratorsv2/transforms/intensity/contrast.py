import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
import numpy as np


class BGContrast():
    def __init__(self, contrast_range):
        self.contrast_range = contrast_range

    def sample_contrast(self, *args, **kwargs):
        if callable(self.contrast_range):
            factor = self.contrast_range()
        else:
            if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
        return factor

    def __call__(self, *args, **kwargs):
        return self.sample_contrast(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + f"(contrast_range={self.contrast_range})"


import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class ContrastTransform(ImageOnlyTransform):
    def __init__(self, contrast_range: RandomScalar, preserve_range: bool, synchronize_channels: bool, p_per_channel: float = 1.0):
        super().__init__()
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = float(p_per_channel)

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        c = img.shape[0]

        # sample on correct device
        apply_idx = (torch.rand(c, device=img.device) < self.p_per_channel).nonzero(as_tuple=False).flatten()
        n = apply_idx.numel()

        if n == 0:
            multipliers = None
        elif self.synchronize_channels:
            m = float(sample_scalar(self.contrast_range, image=img, channel=None))
            multipliers = torch.full((n,), m, device=img.device, dtype=img.dtype)
        else:
            # Still a Python loop because sample_scalar is scalar-by-scalar
            # Use .tolist() to avoid iterating tensor scalars in Python
            ms = [sample_scalar(self.contrast_range, image=img, channel=int(ch)) for ch in apply_idx.tolist()]
            multipliers = torch.as_tensor(ms, device=img.device, dtype=img.dtype)

        return {"apply_to_channel": apply_idx, "multipliers": multipliers}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        idx = params["apply_to_channel"]
        multipliers = params["multipliers"]
        if multipliers is None or idx.numel() == 0:
            return img

        if self.preserve_range:
            for i in range(idx.numel()):
                c = int(idx[i])
                m = multipliers[i]

                x = img[c]
                mean = x.mean()
                minm = x.min()
                maxm = x.max()

                x.sub_(mean)
                x.mul_(m)
                x.add_(mean)
                x.clamp_(minm, maxm)
        else:
            for i in range(idx.numel()):
                c = int(idx[i])
                m = multipliers[i]

                x = img[c]
                mean = x.mean()
                x.sub_(mean)
                x.mul_(m)
                x.add_(mean)

        return img


if __name__ == '__main__':
    from time import time
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    mbt = ContrastTransform(BGContrast((0.75, 1.25)).sample_contrast, True, False, p_per_channel=1)

    times_torch = []
    for _ in range(100):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform

    gnt_bg = ContrastAugmentationTransform((0.75, 1.25), preserve_range=True, per_channel=True, p_per_channel=1)
    times_bg = []
    for _ in range(100):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))
