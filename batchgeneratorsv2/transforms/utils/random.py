from typing import List

import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import numpy as np


class RandomTransform(BasicTransform):
    def __init__(self, transform: BasicTransform, apply_probability: float = 1):
        super().__init__()
        self.transform = transform
        self.apply_probability = apply_probability

    def get_parameters(self, **data_dict) -> dict:
        return {"apply_transform": torch.rand(1).item() < self.apply_probability}

    def apply(self, data_dict: dict, **params) -> dict:
        if params['apply_transform']:
            return self.transform(**data_dict)
        else:
            return data_dict
    
    def __repr__(self):
        ret_str = f"{type(self).__name__}(p={self.apply_probability}, transform={self.transform})"
        return ret_str


class OneOfTransform(BasicTransform):
    """
    Randomly selects and applies one transform from the provided list.

    Each transform must be a callable (usually a BasicTransform subclass).
    This does not override the internal probabilities of the transforms themselves.

    Args:
        list_of_transforms (List[BasicTransform]): A list of transform instances to choose from.
    """

    def __init__(self, list_of_transforms: List[BasicTransform]):
        super().__init__()
        self.list_of_transforms = list_of_transforms

    def __call__(self, **data_dict) -> dict:
        chosen_transform = np.random.choice(self.list_of_transforms)
        return chosen_transform(**data_dict)