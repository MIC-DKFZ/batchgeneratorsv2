import numpy as np
import torch
from typing import Union, Tuple
from scipy.ndimage import median_filter

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class MedianFilterTransform(ImageOnlyTransform):
    """
    Applies a median filter to selected image channels.

    Attributes:
        filter_size (int or Tuple[int, int]): Either fixed filter size or range for random sampling.
        p_same_for_each_channel (float): Probability that all channels share the same filter size.
        p_per_channel (float): Probability of applying the filter to a given channel.
    """

    def __init__(self,
                 filter_size: Union[int, Tuple[int, int]],
                 p_same_for_each_channel: float = 0.0,
                 p_per_channel: float = 1.0):
        super().__init__()
        self.filter_size = filter_size
        self.p_same_for_each_channel = p_same_for_each_channel
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        use_same = np.random.rand() < self.p_same_for_each_channel

        if isinstance(self.filter_size, int):
            sizes = [self.filter_size] * C
        elif use_same:
            sampled_size = int(np.random.randint(*self.filter_size))
            sizes = [sampled_size] * C
        else:
            sizes = [int(np.random.randint(*self.filter_size)) for _ in range(C)]

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        return {
            'filter_sizes': sizes,
            'apply_channel': apply_channel
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        img_np = img.cpu().numpy()
        for c, (apply, size) in enumerate(zip(params['apply_channel'], params['filter_sizes'])):
            if apply:
                img_np[c] = median_filter(img_np[c], size=size)
        return torch.from_numpy(img_np).to(img.device)
