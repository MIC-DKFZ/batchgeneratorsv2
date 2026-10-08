import math
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import fourier_gaussian
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.cropping import crop_tensor


SEG_TIEBREAKS = ('nearest', 'lowest', 'highest')


class SpatialTransform(BasicTransform):
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, List[int], Tuple[int, ...]],
                 random_crop: bool,
                 p_elastic_deform: float = 0,
                 elastic_deform_scale: RandomScalar = (0, 0.2),
                 elastic_deform_magnitude: RandomScalar = (0, 0.2),
                 p_synchronize_def_scale_across_axes: float = 0,
                 p_rotation: float = 0,
                 rotation: RandomScalar = (0, 2 * np.pi),
                 p_rot_per_axis: float = 1,
                 p_scaling: float = 0,
                 scaling: RandomScalar = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 0,
                 bg_style_seg_sampling: bool = False,
                 mode_seg: str = 'bilinear',
                 border_mode_seg: str = "zeros",
                 center_deformation: bool = True,
                 mode_image: str = 'bilinear',
                 padding_mode_image: str = "zeros",
                 padding_value_seg: float = 0,
                 padding_value_image: float = 0,
                 align_corners: bool = False,
                 *,
                 seg_tiebreak: str = 'nearest'
                 ):
        """
        magnitude must be given in pixels!
        deformation scale is given as a paercentage of the edge length

        padding_mode_image: see torch grid_sample documentation. This currently applies to image and regression target
        because both call self._apply_to_image. Can be "zeros", "constant", "reflection", "border". "constant" pads
        with padding_value_image and resamples the padded image, exactly as "zeros" does with 0.

        border_mode_seg: can be "zeros", "constant", "reflection", "border". padding values are only considered for
        the corresponding "constant" modes. "zeros" and "constant" pad the segmentation with a label (0 or
        padding_value_seg) and resample the padded segmentation, so the image ends at the outer edge of its
        border pixels: a sample point up to half a pixel past the last pixel centre still gets that pixel's
        label, and one further out gets the padding label wherever the padding outweighs the image.

        seg_tiebreak: how to settle voxels where two labels share the top interpolated score, which is what
        every boundary voxel becomes when a sample point falls exactly between two labels. 'nearest' takes
        the nearest neighbour label (symmetric in the labels, and the only option that is a function of the
        geometry rather than of the label values), 'lowest' keeps the smallest label (plain argmax, matches
        nnU-Net's resample_torch), 'highest' keeps the largest. Only used when mode_seg != 'nearest'.
        """
        super().__init__()
        self.patch_size = patch_size
        if not isinstance(patch_center_dist_from_border, (tuple, list)):
            patch_center_dist_from_border = [patch_center_dist_from_border] * len(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.random_crop = random_crop
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale  # sigma for blurring offsets, in % of patch size. Larger values mean coarser deformation
        self.elastic_deform_magnitude = elastic_deform_magnitude  # determines the maximum displacement, measured in pixels!!
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_rot_per_axis = p_rot_per_axis
        self.p_scaling = p_scaling
        self.scaling = scaling  # larger numbers = smaller objects!
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes
        self.p_synchronize_def_scale_across_axes = p_synchronize_def_scale_across_axes
        self.bg_style_seg_sampling = bg_style_seg_sampling
        if seg_tiebreak not in SEG_TIEBREAKS:
            raise ValueError(f'unknown seg_tiebreak: {seg_tiebreak}. Must be one of {SEG_TIEBREAKS}')
        self.seg_tiebreak = seg_tiebreak
        self.mode_seg = mode_seg
        self.border_mode_seg = border_mode_seg
        self.center_deformation = center_deformation
        self.mode_image = mode_image
        self.padding_mode_image = padding_mode_image
        self.padding_value_seg = padding_value_seg
        self.padding_value_image = padding_value_image
        self.align_corners = align_corners
        self._grid_cache = {}  # key: (patch_size, dtype) -> base grid tensor

    def _get_base_grid_clone(self) -> torch.Tensor:
        key = tuple(self.patch_size)
        g = self._grid_cache.get(key)
        if g is None:
            g = _create_centered_identity_grid2(self.patch_size).float().contiguous()
            self._grid_cache[key] = g
        return g.clone()

    @staticmethod
    def _get_crop_pad_settings(padding_mode: str, padding_value: float):
        if padding_mode == 'reflection':
            return 'reflect', {}
        if padding_mode == 'border':
            return 'replicate', {}
        if padding_mode == 'zeros':
            return 'constant', {'value': 0}
        if padding_mode == 'constant':
            return 'constant', {'value': padding_value}
        raise RuntimeError(f'Unknown pad mode: {padding_mode}')

    @staticmethod
    def _get_grid_sample_padding_mode(padding_mode: str) -> str:
        if padding_mode in ('zeros', 'constant'):
            return 'zeros'
        if padding_mode in ('border', 'reflection'):
            return padding_mode
        raise RuntimeError(f'Unknown pad mode: {padding_mode}')

    def _kernel_bound(self, spatial_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Per grid axis, the largest |grid coordinate| at which the self.mode_seg kernel still draws on pixels
        inside the image only (bilinear reaches one pixel, bicubic two). Axis k of the grid is spatial axis
        dim - 1 - k, which is how grid_sample reads it (x is the last spatial axis).
        """
        margin = {'bilinear': 0, 'bicubic': 1}[self.mode_seg]
        size = np.array(spatial_shape[::-1], dtype=float)
        # pixel coordinate x < margin or x > size - 1 - margin, as a bound on |grid coordinate| (symmetric)
        if self.align_corners:
            return 1 - 2 * margin / np.maximum(size - 1, 1e-12)
        return 1 - (2 * margin + 1) / size

    def _may_reach_outside(self, grid: torch.Tensor, spatial_shape: Tuple[int, ...]) -> bool:
        """False guarantees that no kernel reaches outside. One contiguous pass over the grid, under 1 ms at 128^3."""
        bound = self._kernel_bound(spatial_shape).min()
        gmin, gmax = torch.aminmax(grid)
        return not (-gmin.item() <= bound and gmax.item() <= bound)

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1

        do_rotation = np.random.uniform() < self.p_rotation
        do_scale = np.random.uniform() < self.p_scaling
        do_deform = np.random.uniform() < self.p_elastic_deform

        if do_rotation:
            angles = [sample_scalar(self.rotation, image=data_dict['image'], dim=i) for i in range(0, dim)]
            if self.p_rot_per_axis < 1:
                for i in range(dim):
                    if np.random.uniform() > self.p_rot_per_axis:
                        angles[i] = 0
        else:
            angles = [0] * dim
        if do_scale:
            if np.random.uniform() <= self.p_synchronize_scaling_across_axes:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=None)] * dim
            else:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=i) for i in range(0, dim)]
        else:
            scales = [1] * dim

        # affine matrix
        if do_scale or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, scales)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], scales)
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
                    for i in range(dim)
                ]

            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, self.patch_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], patch_size=self.patch_size,
                              dim=i, deformation_scale=deformation_scales[i])
                for i in range(dim)]
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

                # tmp = offsets[d].numpy().astype(np.float64)
                # gaussian_filter(tmp, sigmas[d], 0, output=tmp)
                # offsets[d] = torch.from_numpy(tmp).to(offsets.dtype)
                # print(offsets.dtype)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            spatial_dims = tuple(list(range(1, dim + 1)))
            offsets = torch.permute(offsets, (*spatial_dims, 0))
        else:
            offsets = None

        shape = data_dict['image'].shape[1:]
        if not self.random_crop:
            center_location_in_pixels = [i / 2 for i in shape]
        else:
            center_location_in_pixels = []
            for d in range(0, dim):
                mn = self.patch_center_dist_from_border[d]
                mx = shape[d] - self.patch_center_dist_from_border[d]
                if mx < mn:
                    center_location_in_pixels.append(shape[d] / 2)
                else:
                    center_location_in_pixels.append(np.random.uniform(mn, mx))
        # Precompute the deformed grid once (shared by image, segmentation, regression target)
        if affine is not None or offsets is not None:
            grid = self._get_base_grid_clone()

            # we deform first, then rotate
            if offsets is not None:
                grid += offsets
            if affine is not None:
                # grid stores spatial vectors as row vectors (shape [..., dim]), whereas affine is built to act on
                # column vectors (x' = affine @ x). We therefore multiply by affine.T so that each grid point is
                # transformed by affine rather than its transpose (see issue #24).
                grid = torch.matmul(grid, torch.from_numpy(affine.T).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if self.center_deformation and offsets is not None:
                mn = grid.mean(dim=list(range(len(shape))))
            else:
                mn = 0

            new_center = torch.Tensor([c - s / 2 for c, s in zip(center_location_in_pixels, shape)])
            grid += (new_center - mn)
            grid = _convert_my_grid_to_grid_sample_grid(grid, shape)
        else:
            grid = None

        return {
            'center_location_in_pixels': center_location_in_pixels,
            'grid': grid,
            # we don't need them but we keep them so that we can debug better
            'affine': affine,
            'elastic_offsets': offsets,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params['grid'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            pad_mode, pad_kwargs = self._get_crop_pad_settings(self.padding_mode_image, self.padding_value_image)
            return crop_tensor(
                img,
                [math.floor(i) for i in params['center_location_in_pixels']],
                self.patch_size,
                pad_mode=pad_mode,
                pad_kwargs=pad_kwargs,
            )

        grid = params['grid']
        # 'constant' pads with padding_value_image and resamples the padded image, like 'zeros' does with 0 and
        # like the segmentation path does with its padding label. grid_sample only pads with zeros, and it is
        # linear in its input, so sampling (img - v) and adding v back is exactly constant padding with v.
        shift = float(self.padding_value_image) if self.padding_mode_image == 'constant' else 0.
        result = grid_sample(
            (img - shift)[None] if shift != 0 else img[None],
            grid[None],
            mode=self.mode_image,
            padding_mode=self._get_grid_sample_padding_mode(self.padding_mode_image),
            align_corners=self.align_corners,
        )[0]
        if shift != 0:
            result += shift
        return result

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        segmentation = segmentation.contiguous()
        if params['grid'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            pad_mode, pad_kwargs = self._get_crop_pad_settings(self.border_mode_seg, self.padding_value_seg)
            return crop_tensor(
                segmentation,
                [math.floor(i) for i in params['center_location_in_pixels']],
                self.patch_size,
                pad_mode=pad_mode,
                pad_kwargs=pad_kwargs,
            )

        grid = params['grid']
        grid_sample_padding_mode = self._get_grid_sample_padding_mode(self.border_mode_seg)
        # 'zeros' and 'constant' pad the segmentation with a label: 0, or padding_value_seg. Outside the image is
        # then treated exactly like pixels of that label - pad first, then resample - so the image ends at the
        # edge of its outermost pixels, not at their centres. A sample point half a pixel past the last pixel
        # centre still belongs to that pixel, and an augmented patch reaching past the crop does not cut the
        # segmentation short. That is also what the image path does with zeros padding. 'border' and
        # 'reflection' never leave the image.
        pads = grid_sample_padding_mode == 'zeros'
        pad_label = self.padding_value_seg if self.border_mode_seg == 'constant' else 0

        if self.mode_seg == 'nearest':
            # grid_sample only pads with zeros; shifting by the padding label makes that padding the label
            shift = float(pad_label) if pads else 0.
            result_seg = (grid_sample(
                segmentation[None].float() - shift,
                grid[None],
                mode=self.mode_seg,
                padding_mode=grid_sample_padding_mode,
                align_corners=self.align_corners
            )[0] + shift).to(segmentation.dtype)
        else:
            # Both branches compute the same thing: the per-voxel argmax over the interpolated one-hot label
            # channels, with self.seg_tiebreak settling voxels where the top score is shared. They are
            # bit-identical (tests/test_seg_sampling.py pins that) and differ only in time and memory:
            #
            #   bg_style_seg_sampling=True   materializes the (n_labels, *patch_size) float16 score stack and
            #                                argmaxes it in one go.
            #   bg_style_seg_sampling=False  keeps a single running score volume and folds the argmax into the
            #                                label loop.
            #
            # False is the default because it is the better of the two on both counts. Measured on a 64^3
            # patch: 0.51x the time at 3 labels, 0.68x at 12, 0.71x at 40, against 0.5 MB of score buffer
            # instead of 1.5 / 6 / 20 MB. The stack only looked cheaper while the old code was allowed to
            # skip the background label and to answer the two-label case with a single grid_sample - the
            # first is not compatible with an argmax, and the second now sits in front of both branches,
            # where it makes them identical at two labels. True is kept only so callers that pass it
            # explicitly keep working; there is no longer a reason to choose it.
            #
            # The float16 score representation and scale_factor are load-bearing for that bit-identity: float16
            # has a spacing of 0.5 at this magnitude, so it collapses near ties into exact ones, and both
            # branches have to collapse the same ones.
            #
            # Neither branch uses the old `interpolated >= 0.5` assignment. That rule could leave a voxel
            # unwritten - where three or more labels meet, no single indicator has to reach 0.5 - and the voxel
            # then kept the zero result_seg was initialized with, which need not be a label of the input at all.
            scale_factor = 1000
            half = 0.5 * scale_factor
            result_seg = torch.zeros((segmentation.shape[0], *self.patch_size), dtype=segmentation.dtype)
            nn_seg = None  # nearest neighbour sample, computed on first use

            def _nearest():
                # the nearest neighbour of the padded segmentation, i.e. what mode_seg='nearest' returns. Where it
                # is not one of the tied labels it is not used (see use_nn), so it can never invent a label.
                nonlocal nn_seg
                if nn_seg is None:
                    shift = float(pad_label) if pads else 0.
                    nn_seg = (grid_sample(
                        segmentation[None].float() - shift,
                        grid[None],
                        mode='nearest',
                        padding_mode=grid_sample_padding_mode,
                        align_corners=self.align_corners
                    )[0] + shift).to(segmentation.dtype)
                return nn_seg

            # The padding label enters the argmax only for patches whose kernel reaches outside the image, which
            # _may_reach_outside establishes cheaply; the usual patch stays inside, and its scores are computed
            # exactly as they would be without padding.
            may_reach_outside = pads and self._may_reach_outside(grid, segmentation.shape[1:])
            # every label's indicator is built into the same input-sized float buffer (torch.eq writes 0. / 1.
            # straight into it) rather than allocating a bool and a float tensor afresh per label: one pass over
            # the input per label instead of two. Allocated on first use, so single-label channels never need it
            ind_buf = []

            for c in range(segmentation.shape[0]):
                labels = torch.from_numpy(np.sort(pd.unique(segmentation[c].numpy().ravel())))
                if may_reach_outside:
                    labels = torch.unique(torch.cat((labels, torch.tensor([pad_label]).to(labels.dtype))))
                if len(labels) == 1:
                    result_seg[c] = labels[0].item()
                    continue

                def _score(u):
                    if not ind_buf:
                        ind_buf.append(torch.empty(segmentation.shape[1:], dtype=torch.float32))
                    ind = ind_buf[0]
                    torch.eq(segmentation[c], u, out=ind)
                    if may_reach_outside and u == pad_label:
                        # the padding label's indicator is 1 outside the image: sample (ind - 1) with zeros padding
                        # and add the 1 back
                        return grid_sample(
                            ind.sub_(1).mul_(scale_factor)[None, None],
                            grid[None],
                            mode=self.mode_seg,
                            padding_mode=grid_sample_padding_mode,
                            align_corners=self.align_corners
                        )[0][0].add_(scale_factor).to(torch.float16)
                    return grid_sample(
                        ind.mul_(scale_factor)[None, None],
                        grid[None],
                        mode=self.mode_seg,
                        padding_mode=grid_sample_padding_mode,
                        align_corners=self.align_corners
                    )[0][0].to(torch.float16)

                if len(labels) == 2:
                    # Shared by both branches, so they cannot drift apart here. grid_sample is linear in its
                    # input and does not clip, and with the padding label counted the indicators partition the
                    # padded image, so the two scores sum to scale_factor everywhere: the second label is the
                    # argmax exactly where its own score passes half of it, and one grid_sample answers the whole
                    # channel instead of two.
                    hi = _score(labels[1])
                    better = (hi >= half) if self.seg_tiebreak == 'highest' else (hi > half)
                    # with two labels the nearest neighbour is always one of them, so every tie is its to settle
                    use_nn = (hi == half) if self.seg_tiebreak == 'nearest' else None
                    result_seg[c] = labels[0].item()
                    result_seg[c][better] = labels[1]
                elif self.bg_style_seg_sampling:
                    scores = torch.empty((len(labels), *self.patch_size), dtype=torch.float16)
                    for i, u in enumerate(labels):
                        scores[i] = _score(u)
                    if self.seg_tiebreak == 'highest':
                        # argmax reports the first maximum, so reverse the stack to get the last one
                        winner = len(labels) - 1 - scores.flip(0).argmax(0)
                    else:
                        winner = scores.argmax(0)
                    result_seg[c] = labels[winner].to(result_seg.dtype)
                    if self.seg_tiebreak == 'nearest':
                        # The nearest neighbour settles a tie only if its label is one of the tied ones: where
                        # three or more labels meet it can be a label that lost.
                        own = scores.gather(0, torch.searchsorted(labels, _nearest()[c])[None])[0]
                        use_nn = own == scores.max(0).values
                        del own
                    else:
                        use_nn = None
                    del scores
                else:
                    best_val = None
                    if self.seg_tiebreak == 'nearest':
                        # The nearest neighbour's label settles a voxel where it has the top score: where it
                        # won outright that changes nothing, where it is tied it settles the tie. Where three
                        # or more labels meet it can be a label that lost, and then the argmax winner stands.
                        # So keep the score of the nearest neighbour's label per voxel and compare it with the
                        # top score once, at the end. -inf never equals a score: a voxel whose nearest
                        # neighbour is not among the labels keeps the argmax.
                        nn_c = _nearest()[c]
                        nn_score = torch.full(nn_c.shape, float('-inf'), dtype=torch.float16)
                        is_nn = torch.empty(nn_c.shape, dtype=torch.bool)
                    else:
                        nn_c = None
                    better = torch.empty(tuple(self.patch_size), dtype=torch.bool)
                    for u in labels:
                        cur = _score(u)
                        if nn_c is not None:
                            torch.eq(nn_c, u, out=is_nn)
                            torch.where(is_nn, cur, nn_score, out=nn_score)
                        if best_val is None:
                            best_val = cur
                            result_seg[c] = u.item()
                            continue
                        # '>=' lets the later (larger) label take ties, '>' leaves them with the earlier one
                        (torch.ge if self.seg_tiebreak == 'highest' else torch.gt)(cur, best_val, out=better)
                        torch.maximum(best_val, cur, out=best_val)
                        result_seg[c][better] = u
                    use_nn = (nn_score == best_val) if nn_c is not None else None

                if use_nn is not None and torch.any(use_nn):
                    # where the nearest neighbour's label won outright this changes nothing. torch.where rather
                    # than indexing: use_nn holds nearly everywhere, and indexing materializes an index per voxel
                    torch.where(use_nn, _nearest()[c], result_seg[c], out=result_seg[c])

        del grid
        return result_seg.contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError
    

def create_affine_matrix_3d(rotation_angles, scaling_factors):
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
    return RS


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S
    return RS


def _create_centered_identity_grid2(size: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    return grid


def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Union[Tuple[int, ...], List[int]]):
    # rescale
    for d in range(len(original_shape)):
        s = original_shape[d]
        my_grid[..., d] /= (s / 2)
    my_grid = torch.flip(my_grid, (len(my_grid.shape) - 1,))
    # my_grid = my_grid.flip((len(my_grid.shape) - 1,))
    return my_grid


if __name__ == '__main__':
    torch.set_num_threads(1)

    shape = (128, 128, 128)
    patch_size = (128, 128, 128)
    labels = 2


    # seg = torch.rand([i // 32 for i in shape]) * labels
    # seg_up = torch.round(torch.nn.functional.interpolate(seg[None, None], size=shape, mode='trilinear')[0],
    #                      decimals=0).to(torch.int16)
    # img = torch.ones((1, *shape))
    # img[tuple([slice(img.shape[0])] + [slice(i // 4, i // 4 * 2) for i in shape])] = 200


    import SimpleITK as sitk
    # img = camera()
    # seg = None
    img = sitk.GetArrayFromImage(sitk.ReadImage('/media/isensee/raw_data/nnUNet_raw/Dataset226_BraTS2024-BraTS-GLI/imagesTr/BraTS-GLI-00005-100_0001.nii.gz'))
    seg = sitk.GetArrayFromImage(sitk.ReadImage('/media/isensee/raw_data/nnUNet_raw/Dataset226_BraTS2024-BraTS-GLI/labelsTr/BraTS-GLI-00005-100.nii.gz'))

    patch_size = (192, 192, 192)
    sp = SpatialTransform(
        patch_size=(192, 192, 192),
        patch_center_dist_from_border=[i / 2 for i in patch_size],
        random_crop=True,
        p_elastic_deform=0,
        elastic_deform_magnitude=(0.1, 0.1),
        elastic_deform_scale=(0.1, 0.1),
        p_synchronize_def_scale_across_axes=0.5,
        p_rotation=1,
        rotation=(-30 / 360 * np.pi, 30 / 360 * np.pi),
        p_scaling=1,
        scaling=(0.75, 1),
        p_synchronize_scaling_across_axes=0.5,
        bg_style_seg_sampling=True,
        mode_seg='bilinear'
    )

    data_dict = {'image': torch.from_numpy(deepcopy(img[None])).float()}
    if seg is not None:
        data_dict['segmentation'] = torch.from_numpy(deepcopy(seg[None]))
    # out = sp(**data_dict)
    #
    # view_batch(out['image'], out['segmentation'])

    from time import time
    times = []
    for _ in range(10):
        data_dict = {'image': torch.from_numpy(deepcopy(img[None])).float()}
        if seg is not None:
            data_dict['segmentation'] = torch.from_numpy(deepcopy(seg[None]))
        st = time()
        out = sp(**data_dict)
        times.append(time() - st)
    print(np.median(times))


    #################
    # with this part we can qualitatively test that the correct axes are ebing augmented. Just set one of the probs to 1 and off you go
    #################

    # def eldef_scale(image, dim, patch_size):
    #     return 0.1
    #
    # def eldef_magnitude(image, dim, patch_size, deformation_scale):
    #     return 10 if dim == 2 else 0
    #
    # def rot(image, dim):
    #     return 45/360 * 2 * np.pi if dim == 0 else 0
    #
    # def scaling(image, dim):
    #     return 0.5 if dim == 0 else 1
    #
    # # lines
    # patch = torch.zeros((1, 64, 60, 68))
    # patch[:, :, 10, 30] = 1
    # patch[:, 50, :, 30] = 1
    # patch[:, 40, 20, :] = 1
    #
    # # patch_block
    # patch_block = torch.zeros((1, 64, 60, 68))
    # patch_block[:, 22:42, 20:40, 24:44] = 1
    #
    # patch_line = torch.zeros((1, 64, 60, 128))
    # patch_line[:, 22:24, 30:32, 10:-10] = 1
    # use = patch_line
    #
    # sp = SpatialTransform(
    #     patch_size=patch.shape[1:],
    #     patch_center_dist_from_border=0,
    #     random_crop=False,
    #     p_elastic_deform=0,
    #     p_rotation=1,
    #     p_scaling=0,
    #     elastic_deform_scale=eldef_scale,
    #     elastic_deform_magnitude=eldef_magnitude,
    #     p_synchronize_def_scale_across_axes=0,
    #     rotation=rot,
    #     scaling=scaling,
    #     p_synchronize_scaling_across_axes=0,
    #     bg_style_seg_sampling=False,
    #     mode_seg='bilinear'
    # )
    #
    #
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(use[0].numpy()), 'orig.nii.gz')
    #
    # params = sp.get_parameters(image=use)
    # transformed = sp._apply_to_image(use, **params)
    #
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed[0].numpy()), 'transformed.nii.gz')

    # p = torch.zeros((1, 1, 8, 16, 32))
    # p[:, :, 2:6, 10:16, 10:24] = 1
    # grid = _create_identity_grid(p.shape[2:])
    # grid[:, :, :, 0] *= 0.5
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # torch.all(out == p)
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(p[0, 0].numpy()), 'orig.nii.gz')
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(out[0, 0].numpy()), 'transformed.nii.gz')

    #################
    # with this part I verify that the crop through spatialtransforms grid sample yields the same result as crop_tensor
    #################

    # sp = SpatialTransform(
    #     patch_size=(48, 52, 54),
    #     patch_center_dist_from_border=0,
    #     random_crop=True,
    #     p_elastic_deform=0,
    #     p_rotation=1,
    #     p_scaling=0,
    #     rotation=0
    # )
    # sp2 = SpatialTransform(
    #     patch_size=(48, 52, 54),
    #     patch_center_dist_from_border=0,
    #     random_crop=True,
    #     p_elastic_deform=0,
    #     p_rotation=0,
    #     p_scaling=0,
    # )
    #
    # patch = torch.zeros((1, 64, 60, 68))
    # patch[:, :, 10, 30] = 1
    # patch[:, 50, :, 30] = 1
    # patch[:, 40, 20, :] = 1
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(patch[0].numpy()), 'orig.nii.gz')
    #
    # center_coords = [50, 10, 16]
    # params = sp.get_parameters(image=patch)
    # params['center_location_in_pixels'] = center_coords
    # params2 = sp2.get_parameters(image=patch)
    # params2['center_location_in_pixels'] = center_coords
    # transformed = sp._apply_to_image(patch, **params)
    # transformed2 = sp._apply_to_image(patch, **params2)
    #
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed[0].numpy()), 'transformed.nii.gz')
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed2[0].numpy()), 'transformed2.nii.gz')



    ####################
    # This is exploraroty code to check how to retrieve coordinates. I used it to verify that grid_sample does in fact
    # use coordinates in reversed dimension order (zyx and not xyz)
    ####################
    # # create a dummy input which has a unique shape in each exis
    # p = torch.zeros((1, 1, 8, 16, 32))
    # # set one pixel to 1
    # p[:, :, 4, 0, 31] = 1
    # # now create an identity grid. I have verified that this grid yields the same image as the input when used in grid_sample. So the grid is correct
    # grid = _create_identity_grid((8, 16, 32)).contiguous() # grid is shape torch.Size([8, 16, 32, 3])
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # assert torch.all(out == p)  # this passes
    # # reduce the grid to the location we are interested in. That are the coordinates where we placed the 1. The 4:5 etc is only so that we keep the number of dimensions
    # grid = grid[4:5, 0:1, 31:32]
    # # What coordinate would we expect? Note that grid is [-1, 1]
    # # For the first dimension, coordinate 4 out of shape 8 is approximately in the middle, so about 0
    # # For the second dimension, coordinate 0 out of shape 16 is very low, so we expect -1 ish (remember there is aligned corners and shit)
    # # For the third dimension, coordinate 31 out of shape 32 is very high, so we expect 1 ish (remember there is aligned corners and shit)
    # # So we expect [0, -1, 1]
    # # What do we get?
    # print(grid)
    # # > tensor([[[[ 0.9688, -0.9375,  0.1250]]]])
    # # not what we expect
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # assert out.item() == 1
