from typing import Optional
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class GammaTransform(ImageOnlyTransform):
    def __init__(self,
                 gamma: RandomScalar,
                 p_invert_image: float,
                 synchronize_channels: bool,
                 p_per_channel: float,
                 p_retain_stats: float):
        super().__init__()
        self.gamma = gamma
        self.p_invert_image = float(p_invert_image)
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = float(p_per_channel)
        self.p_retain_stats = float(p_retain_stats)

    def get_parameters(self, **data_dict) -> dict:
        img: torch.Tensor = data_dict["image"]
        c = img.shape[0]
        device = img.device
        dtype = img.dtype

        apply_idx = (torch.rand(c, device=device) < self.p_per_channel).nonzero(as_tuple=False).flatten()
        n = apply_idx.numel()
        if n == 0:
            return {"apply_to_channel": apply_idx,
                    "retain_stats": None,
                    "invert_image": None,
                    "gamma": None}

        retain_stats = (torch.rand(n, device=device) < self.p_retain_stats)
        invert_image = (torch.rand(n, device=device) < self.p_invert_image)

        if self.synchronize_channels:
            g = float(sample_scalar(self.gamma, image=img, channel=None))
            gamma = torch.full((n,), g, device=device, dtype=dtype)
        else:
            # sample_scalar is scalar-based; keep loop but avoid tensor scalar iteration
            gs = [float(sample_scalar(self.gamma, image=img, channel=int(ch))) for ch in apply_idx.tolist()]
            gamma = torch.as_tensor(gs, device=device, dtype=dtype)

        return {
            "apply_to_channel": apply_idx,
            "retain_stats": retain_stats,
            "invert_image": invert_image,
            "gamma": gamma,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        idx: torch.Tensor = params["apply_to_channel"]
        if idx.numel() == 0:
            return img

        retain_stats: torch.Tensor = params["retain_stats"]
        invert_image: torch.Tensor = params["invert_image"]
        gamma: torch.Tensor = params["gamma"]

        # constants
        eps = 1e-7

        # Loop over selected channels (good for small C)
        for k in range(idx.numel()):
            c = int(idx[k])
            r = bool(retain_stats[k])
            inv = bool(invert_image[k])
            g = gamma[k]

            x = img[c]

            if inv:
                x.mul_(-1)

            if r:
                mean = x.mean()
                std = x.std()

            minm = x.min()
            maxm = x.max()
            rnge = maxm - minm
            denom = torch.clamp(rnge, min=eps)

            # In-place gamma: x = (((x - min) / denom) ** g) * rnge + min
            x.sub_(minm)
            x.div_(denom)
            x.pow_(g)
            x.mul_(rnge)
            x.add_(minm)

            if r:
                mn_here = x.mean()
                std_here = x.std()
                x.sub_(mn_here)
                x.mul_(std / torch.clamp(std_here, min=eps))
                x.add_(mean)

            if inv:
                x.mul_(-1)

        return img



if __name__ == '__main__':
    from time import time
    import numpy as np
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    mbt = GammaTransform((0.7, 1.5), 0, False, 1, 1)

    times_torch = []
    for _ in range(100):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.color_transforms import GammaTransform as BGGamma

    gnt_bg = BGGamma((0.7, 1.5), False, True, retain_stats=True, p_per_sample=1)
    times_bg = []
    for _ in range(100):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))
