import math
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import fourier_gaussian
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.spatial.spatial import _create_centered_identity_grid2, \
    _convert_my_grid_to_grid_sample_grid, create_affine_matrix_2d, create_affine_matrix_3d
from batchgeneratorsv2.transforms.utils.cropping import crop_tensor


class ChannelMisalignmentTransform(ImageOnlyTransform):
    """
    The misalignment data augmentation is introduced in Nature Scientific reports 2023.

    Apply channel-wise misalignment to selected image channels.
    This transform simulates registration errors between channels by randomly
    applying one or more of the following operations to the specified image
    channels:
    - squeezing/scaling (good approximation for misalignments between the T2w and DWI MRI sequences)
    - rotation
    - translation via shifted crop center

    If you use this augmentation please cite: https://www.nature.com/articles/s41598-023-46747-z

    Parameters
    ----------
    im_channels_2_misalign : Tuple[int, ...]
        Image channels to which the misalignment is applied.

    squeezing_zyx : Tuple[float, ...], default=(0.1, 0, 0)
        Maximum relative scaling deviation per axis in ZYX order.
        For each active axis, the scale factor is sampled uniformly from [1 - s, 1 + s].

    p_squeeze : float, default=0.0
        Probability of applying squeezing/scaling.

    rotation_ax_cor_sag : Tuple[float, ...], default=(np.pi, np.pi, np.pi)
        Maximum absolute rotation angle per axis in axial/coronal/sagittal
        order. Angles are sampled uniformly from [-a, a].

    rad_or_deg : {"rad", "deg"}
        Unit of `rotation_ax_cor_sag`.

    p_rotation : float, default=0.0
        Probability of applying rotation.

    shift_zyx : Tuple[int, ...], default=(2, 32, 32)
        Maximum integer shift per axis in ZYX order. For each axis, the shift
        is sampled uniformly from [-s, s].

    p_shift : float, default=0.0
        Probability of applying translation.

    """

    def __init__(self,
                 im_channels_2_misalign: Tuple[int,] = [0, ],

                 squeezing_zyx: Tuple[float, ...] = (0.1, 0, 0),
                 p_squeeze: float = 0.0,

                 rotation_ax_cor_sag: Tuple[float, ...] = (np.pi, np.pi, np.pi),
                 rad_or_deg=None,
                 p_rotation: float = 0.0,

                 shift_zyx: Tuple[int, ...] = (2, 32, 32),
                 p_shift: float = 0.0,
                 ):
        super().__init__()
        self.im_channels_2_misalign = im_channels_2_misalign

        self.squeezingZYX = squeezing_zyx
        self.p_squeeze = p_squeeze

        if rad_or_deg == "rad":
            if any(rot > np.pi / 12 for rot in rotation_ax_cor_sag):
                raise Warning("The rotation is probably too big")
            if any(rot > np.pi for rot in rotation_ax_cor_sag):
                raise ValueError("The rotation is probably in deg or bigger than 180°")
            self.rotation_ax_cor_sag = rotation_ax_cor_sag
        elif rad_or_deg == "deg":
            self.rotation_ax_cor_sag = [rot / 360 * (2 * np.pi) for rot in rotation_ax_cor_sag]
        else:
            raise RuntimeError('Please define the rad_or_deg: "rad"/"deg"')
        self.p_rotation = p_rotation

        self.shiftZYX = shift_zyx
        self.p_shift = p_shift

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1

        do_squeeze = np.random.uniform() < self.p_squeeze
        do_rotation = np.random.uniform() < self.p_rotation
        do_shift = np.random.uniform() < self.p_shift
        do_deform = False

        # Squeeze
        if do_squeeze:
            squeezes = [np.random.uniform(1 - self.squeezingZYX[i], 1 + self.squeezingZYX[i]) for i in range(dim)]
        else:
            squeezes = [1] * dim

        # Rotation
        if do_rotation:
            angles = [np.random.uniform(-self.rotation_ax_cor_sag[i], self.rotation_ax_cor_sag[i]) for i in range(dim)]
        else:
            angles = [0] * dim

        # affine matrix
        if do_squeeze or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, squeezes)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], squeezes)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        # elastic deformation. We need to create the displacement field here
        # we use the method from augment_spatial_2 in batchgenerators
        if do_deform:
            if np.random.uniform() <= self.p_synchronize_def_scale_across_axes:
                deformation_scales = [
                                         sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=None,
                                                       patch_size=self.patch_size)
                                     ] * dim
            else:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=i,
                                  patch_size=self.patch_size)
                    for i in range(0, 3)
                ]

            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, self.patch_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], patch_size=self.patch_size,
                              dim=i, deformation_scale=deformation_scales[i])
                for i in range(0, 3)]
            # doing it like this for better memory layout for blurring
            offsets = torch.normal(mean=0, std=1, size=(dim, *self.patch_size))

            # all the additional time elastic deform takes is spent here
            for d in range(dim):
                # fft torch, slower
                # for i in range(offsets.ndim - 1):
                #     offsets[d] = blur_dimension(offsets[d][None], sigmas[d], i, force_use_fft=True, truncate=6)[0]

                # fft numpy, this is faster o.O
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            offsets = torch.permute(offsets, (1, 2, 3, 0))
        else:
            offsets = None

        # shape = data_dict['image'].shape[1:]
        # if do_shift:
        #     for i in shape:
        #         print(i)
        #     center_location_in_pixels = [i / 2 + np.random.randint(self.shiftXYZ[j], self.shiftXYZ[j]+1) for i, j in zip(shape, range(dim - 1, -1, -1))][::-1]
        # else:
        #     center_location_in_pixels = [i / 2 for i in shape][::-1]

        shape = data_dict['image'].shape[1:]
        if not do_shift:
            center_location_in_pixels = [i / 2 for i in shape]
        else:
            center_location_in_pixels = [shape[i] / 2 + np.random.randint(-self.shiftZYX[i], self.shiftZYX[i] + 1) for i
                                         in range(dim)]

        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        im_shape = img.shape[1:]
        if params['affine'] is None and params['elastic_offsets'] is None:
            for ch in self.im_channels_2_misalign:
                img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0),
                                           [math.floor(i) for i in params['center_location_in_pixels']], im_shape,
                                           pad_mode='constant', pad_kwargs={'value': 0})
            return img
        else:
            grid = _create_centered_identity_grid2(im_shape)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(img.ndim - 1)))
            else:
                mn = 0

            # new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])])
            new_center = torch.Tensor([0, 0, 0])
            grid += (new_center - mn)

            for ch in self.im_channels_2_misalign:
                img[ch, ...] = grid_sample(img[ch, ...].unsqueeze(0).unsqueeze(0),
                                           _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                                           mode='bilinear', padding_mode="zeros", align_corners=False)[0]
                img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0),
                                           [math.floor(i) for i in params['center_location_in_pixels']], im_shape,
                                           pad_mode='constant', pad_kwargs={'value': 0})
            return img
