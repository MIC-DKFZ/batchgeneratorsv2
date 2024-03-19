import numpy as np
from typing import Tuple

import torch

from batchgenerators_torch.helpers.scalar_type import ScalarType
from batchgenerators_torch.transforms.base.basic_transform import BasicTransform


class SpatialTransform(BasicTransform):
    def __init__(self, patch_size: Tuple[int, ...], patch_center_dist_from_border: int,
                 random_crop: bool,
                 p_elastic_deform, elastic_deform_scale: ScalarType,
                 p_rotation, rotation: ScalarType,
                 p_scaling, scaling: ScalarType, p_synchronize_scaling_across_axes: float,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.random_crop = random_crop
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_scaling = p_scaling
        self.scaling = scaling
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1
        # affine matrix

        # rotation
        # scaling

        # elastic deformation. We need to create the displacement field here

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    """
    Create a 3x4 affine matrix given rotation angles and scaling factors.

    Parameters:
    - rotation_angles: A tuple or list of three angles (in radians) for rotation around the x, y, and z axes.
    - scaling_factors: A tuple or list of three scaling factors for the x, y, and z axes.

    Returns:
    - A 3x4 numpy array representing the affine transformation matrix.
    """
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = Rz @ Ry @ Rx @ S

    # Create a 3x4 affine matrix (for rotations and scaling, the last column is [0, 0, 0, 1])
    affine_matrix = np.hstack((RS, np.array([[0], [0], [0]])))

    return affine_matrix


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    """
    Create a 2x3 affine matrix for 2D transformations given a rotation angle and scaling factors.

    Parameters:
    - rotation_angle: The angle (in radians) for rotation.
    - scaling_factors: A tuple or list of two scaling factors for the x and y axes.

    Returns:
    - A 2x3 numpy array representing the affine transformation matrix.
    """
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S

    # Since it's 2D, we append a row for homogeneous coordinates to make it 3x3,
    # but we'll only return the first 2 rows for your specific request,
    # making the effective transformation matrix 2x3.
    affine_matrix_2d = np.hstack((RS, np.array([[0], [0]])))

    return affine_matrix_2d

def _create_identity_grid(size: List[int]) -> Tensor:
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    return torch.stack([grid_x, grid_y], -1).unsqueeze(0)  # 1 x H x W x 2

def elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: str = "bilinear",
    fill: Optional[Union[int, float, List[float]]] = None,
) -> Tensor:

    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    size = list(img.shape[-2:])
    displacement = displacement.to(img.device)

    identity_grid = _create_identity_grid(size)
    grid = identity_grid.to(img.device) + displacement
    return _apply_grid_transform(img, grid, interpolation, fill)


def displacement_field(data: torch.Tensor):
    downscaling_global = np.random.uniform() ** 2 * 4 + 2
    # local downscaling can vary a bit relative to global
    granularity_scale_local = np.random.uniform(round(max(downscaling_global - 1.5, 2)),
                                                round(downscaling_global + 1.5), size=3)

    B, _, D, H, W = data.size()
    random_field_size = [round(j / i) for i, j in zip(granularity_scale_local, data.shape[2:])]
    pool_kernel_size = [min(i // 4 * 2 + 1, round(7 / 4 * downscaling_global) // 2 * 2 + 1) for i in
                        random_field_size]  # must be odd
    pool_padding = [(i - 1) // 2 for i in pool_kernel_size]
    aug1 = F.avg_pool3d(
        F.avg_pool3d(
            torch.randn((B, 2, *random_field_size), device=data.device),
            pool_kernel_size, stride=1, padding=pool_padding),
        pool_kernel_size, stride=1, padding=pool_padding)

_apply_grid_transform =
img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)
