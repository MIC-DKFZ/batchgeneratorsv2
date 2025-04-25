import numpy as np
import torch
from typing import Tuple, Set, List

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class Rot90Transform(BasicTransform):
    """
    Applies a random 90-degree rotation to image and associated targets along randomly chosen axes.

    Attributes:
        num_rot (Tuple[int]): Possible multiples of 90 degrees to rotate (e.g., (1, 2, 3)).
        allowed_axes (Set[int]): Spatial axes to randomly select rotation axes from (e.g., {0, 1, 2}).
        p_per_sample (float): Probability of applying the transform to a sample.
    """

    def __init__(self, num_axis_combinations: RandomScalar, num_rot_per_combination: Tuple[int, ...] = (1, 2, 3),
                 allowed_axes: Set[int] = {0, 1, 2}):
        super().__init__()
        self.num_axis_combinations = num_axis_combinations
        self.num_rot_per_combination = num_rot_per_combination
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        n_axes_combinations = round(sample_scalar(self.num_axis_combinations))
        axis_combinations = []
        num_rot_per_combination = []
        for i in range(n_axes_combinations):
            num_rot_per_combination.append(int(np.random.choice(self.num_rot_per_combination)))
            axis_combinations.append(sorted(np.random.choice(list(self.allowed_axes), size=2, replace=False)))
            # +1 because we skip channel dimension
            axis_combinations[-1] = [a + 1 for a in axis_combinations[-1]]
        return {
            'num_rot_per_combination': num_rot_per_combination,
            'axis_combinations': axis_combinations
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(img, **params)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(seg, **params)

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(regression_target, **params)

    def _maybe_rot90(self, tensor: torch.Tensor, num_rot_per_combination: List[int], axis_combinations: List[Tuple[int, int]]) -> torch.Tensor:
        for n_rot, axes in zip(num_rot_per_combination, axis_combinations):
            tensor = torch.rot90(tensor, k=n_rot, dims=axes)
        return tensor

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

if __name__ == '__main__':
    # Create dummy 3D image and segmentation tensors: (C, X, Y, Z)
    image = torch.arange(1 * 8 * 8 * 8).reshape(1, 8, 8, 8).float()
    seg = torch.zeros_like(image)

    # Instantiate the transform
    transform = Rot90Transform(num_axis_combinations=2, num_rot_per_combination=(1, 2, 3), allowed_axes={0, 1, 2})  # always apply for demo

    # Get random parameters for this sample
    params = transform.get_parameters(image=image, segmentation=seg)

    # Apply transform
    image_rot = transform._apply_to_image(image, **params)
    seg_rot = transform._apply_to_segmentation(seg, **params)

    # Print to verify
    print("Original image shape:", image.shape)
    print("Rotated image shape:", image_rot.shape)
    print("Rotation parameters:", params)