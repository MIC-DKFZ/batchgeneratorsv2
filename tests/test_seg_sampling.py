import unittest

import numpy as np
import torch
from torch.nn.functional import grid_sample

from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform


def grid_coord(x, n):
    """normalized grid coordinate that samples exactly at input coordinate x (align_corners=False)"""
    return (2.0 * x + 1.0) / n - 1.0


def blobs(rng, shape, n_labels):
    v = sum(np.roll(rng.rand(*shape), rng.randint(0, 5), axis=rng.randint(0, len(shape))) for _ in range(3))
    q = np.quantile(v, np.linspace(0, 1, n_labels + 1)[1:-1]) if n_labels > 1 else []
    return np.digitize(v, q).astype(np.int16)


def transform(patch_size, tiebreak, bg_style, **kwargs):
    kwargs.setdefault('random_crop', False)
    return SpatialTransform(patch_size=patch_size, patch_center_dist_from_border=0,
                            bg_style_seg_sampling=bg_style, seg_tiebreak=tiebreak, mode_seg='bilinear', **kwargs)


def sample_row(row, tiebreak, bg_style):
    """sample a length-8 label row at x = 0.5, 2.5, 4.5, 6.5, i.e. straight through the 2x downsampling grid"""
    seg = torch.tensor([[row]], dtype=torch.int16)
    grid = torch.tensor([[[grid_coord(x, 8), 0.0] for x in (0.5, 2.5, 4.5, 6.5)]])
    t = transform((1, 4), tiebreak, bg_style)
    return t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None).numpy().ravel()


def padded_argmax(seg, grid, mode, pad_label, tiebreak='lowest', margin=8):
    """
    The definition of 'zeros' / 'constant' padding for a segmentation, computed independently of
    SpatialTransform: pad the channel with `margin` pixels of pad_label, move the grid to the padded image,
    and take the argmax there, where no sample point is outside any more.
    """
    shape = seg.shape[1:]
    padded = torch.nn.functional.pad(seg[0][None].float(), [margin] * (2 * len(shape)), value=float(pad_label))[0]
    g = grid.clone()
    for k, n in enumerate(shape[::-1]):  # grid axis k is spatial axis dim - 1 - k
        x = ((g[..., k] + 1) * n - 1) / 2 + margin
        g[..., k] = (2 * x + 1) / (n + 2 * margin) - 1
    labels = torch.unique(padded).to(seg.dtype)
    scores = torch.stack([grid_sample(((padded == u).float() * 1000)[None, None], g[None], mode=mode,
                                      padding_mode='border', align_corners=False)[0, 0].to(torch.float16)
                          for u in labels])
    if tiebreak == 'highest':
        return labels[len(labels) - 1 - scores.flip(0).argmax(0)]
    return labels[scores.argmax(0)]


class TestSegSampling(unittest.TestCase):
    """
    SpatialTransform resamples a segmentation by interpolating each label's indicator and picking a winner
    per voxel. Both bg_style_seg_sampling branches must implement the same rule - the per-voxel argmax, with
    seg_tiebreak settling exact ties - and must never write a label that is not in the input.
    """

    def test_branches_agree_exactly(self):
        for i in range(30):
            rng = np.random.RandomState(1000 + i)
            seg = blobs(rng, (32, 32), rng.randint(1, 8))
            if rng.rand() < 0.4:  # label values need not start at zero
                seg = seg + rng.randint(1, 4)
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = []
                for bg_style in (True, False):
                    torch.manual_seed(i)
                    np.random.seed(i)
                    t = transform((24, 24), tiebreak, bg_style, random_crop=True, p_elastic_deform=1.0,
                                  p_rotation=1.0, rotation=(0, 2 * np.pi), p_scaling=1.0, scaling=(0.7, 1.4))
                    s = torch.from_numpy(seg)[None]
                    out.append(t(**{'image': s.float(), 'segmentation': s})['segmentation'].numpy())
                self.assertTrue(np.array_equal(out[0], out[1]),
                                f'branches disagree on {np.sum(out[0] != out[1])} voxels (i={i}, {tiebreak})')

    def test_no_label_is_invented(self):
        """
        The regression that motivated the rewrite. The old rule assigned with `interpolated >= 0.5` into a
        zero-initialized result, so a voxel that no label claimed kept the 0 - which need not occur in the
        input. Four labels meeting at a sample point give 0.25 each and no label claims it.
        """
        seg = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int16)
        grid = torch.tensor([[[0.0, 0.0]]])  # the exact centre of the 2x2 block
        for bg_style in (True, False):
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = transform((1, 1), tiebreak, bg_style)._apply_to_segmentation(
                    seg, grid=grid, center_location_in_pixels=None).numpy()
                self.assertTrue(set(np.unique(out).tolist()) <= {1, 2, 3, 4},
                                f'produced a label that is not in the input: {np.unique(out)} '
                                f'(bg_style={bg_style}, {tiebreak})')

    def test_no_label_is_invented_under_augmentation(self):
        # 'border' padding, so no voxel is outside the image: with zeros padding the outside is label 0 by
        # definition, and test_zeros_padding_is_label_zero covers that
        for i in range(20):
            rng = np.random.RandomState(2000 + i)
            seg = blobs(rng, (32, 32), rng.randint(2, 8)) + rng.randint(1, 4)  # never contains 0
            present = set(np.unique(seg).tolist())
            for bg_style in (True, False):
                for tiebreak in ('nearest', 'lowest', 'highest'):
                    torch.manual_seed(i)
                    np.random.seed(i)
                    t = transform((24, 24), tiebreak, bg_style, random_crop=True, p_elastic_deform=1.0,
                                  p_rotation=1.0, rotation=(0, 2 * np.pi), p_scaling=1.0, scaling=(0.7, 1.4),
                                  border_mode_seg='border')
                    s = torch.from_numpy(seg)[None]
                    out = t(**{'image': s.float(), 'segmentation': s})['segmentation'].numpy()
                    self.assertTrue(set(np.unique(out).tolist()) <= present,
                                    f'invented {sorted(set(np.unique(out).tolist()) - present)} '
                                    f'(i={i}, bg_style={bg_style}, {tiebreak})')

    def test_nearest_tiebreak_is_symmetric_in_the_label_values(self):
        """
        Sampling exactly between two labels is an exact tie, and it is not a rare case: it is what an even
        integer downsampling factor does at every boundary voxel. 'nearest' has to answer it from the
        geometry, so relabelling must move the same decision with the labels. 'lowest' and 'highest' answer
        it from the label values, so they do not.
        """
        for lo, hi in ((1, 2), (2, 7), (3, 4)):
            forward = [lo] * 5 + [hi] * 3
            swapped = [hi] * 5 + [lo] * 3
            remap = {lo: hi, hi: lo}
            for bg_style in (True, False):
                a = sample_row(forward, 'nearest', bg_style)
                b = sample_row(swapped, 'nearest', bg_style)
                self.assertTrue(np.array_equal([remap[v] for v in a], list(b)),
                                f"'nearest' is not symmetric for ({lo}, {hi}): {a} vs {b}")
                # the tied voxel is the third one (x = 4.5, exactly between index 4 and index 5)
                self.assertEqual(sample_row(forward, 'lowest', bg_style)[2], lo)
                self.assertEqual(sample_row(forward, 'highest', bg_style)[2], hi)

    def test_nearest_tiebreak_matches_a_nearest_neighbour_sample(self):
        row = [1] * 5 + [2] * 3
        seg = torch.tensor([[row]], dtype=torch.int16)
        grid = torch.tensor([[[grid_coord(x, 8), 0.0] for x in (0.5, 2.5, 4.5, 6.5)]])
        nn = grid_sample(seg[None].float(), grid[None], mode='nearest', padding_mode='border',
                         align_corners=False)[0].numpy().ravel()
        for bg_style in (True, False):
            self.assertTrue(np.array_equal(sample_row(row, 'nearest', bg_style), nn.astype(np.int16)))

    def test_single_label_is_passed_through(self):
        for bg_style in (True, False):
            self.assertTrue(np.array_equal(sample_row([7] * 8, 'nearest', bg_style), np.full(4, 7)))

    def test_3d(self):
        rng = np.random.RandomState(7)
        seg = torch.from_numpy(blobs(rng, (20, 20, 20), 5))[None]
        out = []
        for bg_style in (True, False):
            torch.manual_seed(0)
            np.random.seed(0)
            t = transform((16, 16, 16), 'nearest', bg_style, random_crop=True, p_rotation=1.0,
                          rotation=(0, 2 * np.pi), p_scaling=1.0, scaling=(0.7, 1.4))
            out.append(t(**{'image': seg.float(), 'segmentation': seg})['segmentation'].numpy())
        self.assertTrue(np.array_equal(out[0], out[1]))
        self.assertEqual(out[0].shape, (1, 16, 16, 16))

    def test_zeros_padding_is_label_zero(self):
        """
        With zeros padding the outside of the image is label 0, whatever labels the channel holds, as it is on
        the crop path and with mode_seg='nearest'. The argmax alone would hand a point entirely outside to
        labels[0] (labels[-1] for 'highest'), i.e. to whichever label happens to be the smallest in this
        channel, so the result would depend on which labels are present. Points only partly outside still
        take the argmax over the labels they do see.
        """
        n = 8
        # x = -3 and x = 10 lie entirely outside, x = -0.4 and x = 7.4 in the band that is partly outside
        xs = (-3.0, -0.4, 3.5, 7.4, 10.0)
        grid = torch.tensor([[[grid_coord(x, n), 0.0] for x in xs]])
        for row in ([1] * 4 + [2] * 4, [1] * 3 + [2] * 2 + [3] * 3, [5] * 4 + [9] * 4):
            seg = torch.tensor([[row]], dtype=torch.int16)
            for border_mode in ('zeros', 'constant'):
                for bg_style in (True, False):
                    for tiebreak in ('nearest', 'lowest', 'highest'):
                        t = transform((1, len(xs)), tiebreak, bg_style, border_mode_seg=border_mode,
                                      padding_value_seg=0)
                        out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None).numpy().ravel()
                        msg = f'{row} {border_mode} bg_style={bg_style} {tiebreak}: {out}'
                        self.assertEqual((out[0], out[4]), (0, 0), msg)
                        self.assertEqual((out[1], out[3]), (row[0], row[-1]), msg)
            t = SpatialTransform(patch_size=(1, len(xs)), patch_center_dist_from_border=0, random_crop=False,
                                 mode_seg='nearest')
            out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None).numpy().ravel()
            self.assertEqual((out[0], out[4]), (0, 0))

    def test_two_labels_match_the_full_argmax(self):
        """
        Two labels take a shortcut (one grid_sample, `hi > half`) that is only the argmax where the two scores
        sum to scale_factor. Check it against an explicit argmax with sample points inside, across and beyond
        the border, for every padding mode and for bicubic, whose kernel reaches one pixel further. For 'zeros'
        and 'constant' the reference pads the segmentation for real and resamples the padded image.
        """
        rng = np.random.RandomState(3)
        for i in range(10):
            seg = torch.from_numpy(blobs(rng, (16, 16), 2) + rng.randint(0, 3))[None]
            labels = torch.unique(seg)
            # sample points everywhere, including a margin outside the image
            grid = torch.from_numpy(rng.uniform(-1.3, 1.3, size=(12, 12, 2))).float()
            for padding, value in (('zeros', 0), ('constant', -1), ('border', 0), ('reflection', 0)):
                for mode in ('bilinear', 'bicubic'):
                    t = transform((12, 12), 'lowest', False, border_mode_seg=padding, padding_value_seg=value)
                    t.mode_seg = mode
                    out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None)[0]
                    if padding in ('zeros', 'constant'):
                        ref = padded_argmax(seg, grid, mode, value).to(out.dtype)
                    else:
                        scores = torch.stack([
                            grid_sample(((seg[0] == u).float() * 1000)[None, None], grid[None], mode=mode,
                                        padding_mode=padding, align_corners=False)[0, 0].to(torch.float16)
                            for u in labels])
                        ref = labels[scores.argmax(0)].to(out.dtype)
                    self.assertTrue(torch.equal(out, ref),
                                    f'{int((out != ref).sum())} voxels differ (i={i}, {padding}, {mode})')

    def test_nearest_tiebreak_only_picks_among_the_tied_labels(self):
        """
        Where three or more labels meet, the nearest neighbour can be a label that lost: the centre of this
        2x2x2 block scores 1 and 2 at 0.375 each and 3 at 0.25, and its nearest neighbour is a 3.
        """
        block = torch.tensor([3, 1, 1, 1, 2, 2, 2, 3], dtype=torch.int16).reshape(2, 2, 2)
        seg = block.repeat(4, 4, 4)[None]
        centres = [grid_coord(2 * i + 0.5, 8) for i in range(4)]
        grid = torch.stack(torch.meshgrid(*[torch.tensor(centres)] * 3, indexing='ij')[::-1], dim=-1).float()
        for bg_style in (True, False):
            for tiebreak, expected in (('nearest', 1), ('lowest', 1), ('highest', 2)):
                t = transform((4, 4, 4), tiebreak, bg_style)
                out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None)
                self.assertTrue((out == expected).all(), f'{torch.unique(out).tolist()} ({tiebreak}, {bg_style})')

    def test_winner_has_the_top_score(self):
        """
        Whatever settles a tie, the label a voxel gets must attain the maximum (float16) score there. Random
        labels sampled at the centres of 2x2x2 blocks put three or more labels into an exact tie at many voxels;
        random rotations and scalings of smooth blobs cover the general case.
        """
        rng = np.random.RandomState(8)
        centres = torch.tensor([grid_coord(2 * i + 0.5, 12) for i in range(6)])
        block_grid = torch.stack(torch.meshgrid(*[centres] * 3, indexing='ij')[::-1], dim=-1).float()
        for i in range(12):
            if i % 2:
                seg = torch.from_numpy(rng.randint(0, rng.randint(3, 6), (12, 12, 12)).astype(np.int16))[None]
            else:
                seg = torch.from_numpy(blobs(rng, (12, 12, 12), rng.randint(3, 7)))[None]
            labels = torch.unique(seg)
            for bg_style in (True, False):
                for tiebreak in ('nearest', 'lowest', 'highest'):
                    if i % 2:
                        t = transform((6, 6, 6), tiebreak, bg_style, border_mode_seg='border')
                        grid = block_grid
                    else:
                        torch.manual_seed(i)
                        np.random.seed(i)
                        t = transform((10, 10, 10), tiebreak, bg_style, border_mode_seg='border', p_rotation=1.0,
                                      rotation=(0, 2 * np.pi), p_scaling=1.0, scaling=(0.7, 1.4))
                        grid = t.get_parameters(image=seg.float(), segmentation=seg)['grid']
                    out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None)[0]
                    scores = torch.stack([grid_sample(((seg[0] == u).float() * 1000)[None, None], grid[None],
                                                      mode='bilinear', padding_mode='border',
                                                      align_corners=False)[0, 0].to(torch.float16)
                                          for u in labels])
                    own = scores.gather(0, torch.searchsorted(labels, out)[None])[0]
                    self.assertTrue(torch.equal(own, scores.max(0).values),
                                    f'{int((own != scores.max(0).values).sum())} voxels ({tiebreak}, {bg_style}, i={i})')

    def test_may_reach_outside(self):
        """
        _may_reach_outside decides whether the padding label has to enter the argmax at all. False must
        guarantee that no kernel draws on a pixel outside the image, on a non-cubic image so that pairing a
        grid axis with the wrong spatial axis shows; bicubic reaches one pixel further than bilinear.
        """
        rng = np.random.RandomState(5)
        for shape in ((6, 11), (5, 9, 14)):
            for reach in (0.3, 0.6, 0.8, 0.9, 1.0, 1.2):
                grid = torch.from_numpy(rng.uniform(-reach, reach, size=(7, 8, 9)[:len(shape)] + (len(shape),))).float()
                for align_corners in (False, True):
                    for mode in ('bilinear', 'bicubic'):
                        t = SpatialTransform(patch_size=grid.shape[:-1], patch_center_dist_from_border=0,
                                             random_crop=False, mode_seg=mode, align_corners=align_corners)
                        if len(shape) == 3 and mode == 'bicubic':
                            continue  # grid_sample has no 3D bicubic
                        # a kernel that draws on the padding shows up as weight missing from a sampled ones image
                        w_in = grid_sample(torch.ones((1, 1, *shape)), grid[None], mode=mode, padding_mode='zeros',
                                           align_corners=align_corners)[0, 0]
                        reaches = bool((torch.abs(w_in - 1) > 1e-5).any())
                        if not t._may_reach_outside(grid, shape):
                            self.assertFalse(reaches, f'{shape} {reach} {align_corners} {mode}')
                        if reach > 1.0:  # past +-1 is outside even with align_corners, where +-1 is a pixel centre
                            self.assertTrue(t._may_reach_outside(grid, shape))

    def test_padding_is_a_label(self):
        """
        'zeros' and 'constant' pad the segmentation with a label and resample the padded segmentation. On a row
        of four pixels, x = 3.3 lies inside the last pixel (past its centre), x = 3.7 is 0.2 px beyond the image
        with 70 % of a linear kernel in the padding, and x = 5 is entirely outside. Single-label channels are
        padded too.
        """
        xs = [3.3, 3.7, 5.0, -3.0]
        grid = torch.tensor([[[grid_coord(x, 4), grid_coord(0.5, 2)] for x in xs]]).float()
        for row in ([1, 1, 2, 2], [1, 3, 2, 2], [2, 2, 2, 2]):
            seg = torch.tensor([row] * 2, dtype=torch.int16)[None]
            for border_mode, pad in (('zeros', 0), ('constant', -1)):
                for mode_seg in ('nearest', 'bilinear'):
                    for bg_style in (True, False):
                        t = SpatialTransform((1, 4), 0, False, mode_seg=mode_seg, border_mode_seg=border_mode,
                                             padding_value_seg=pad, bg_style_seg_sampling=bg_style)
                        out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None).ravel()
                        self.assertEqual(out.tolist(), [2, pad, pad, pad], f'{row} {border_mode} {mode_seg} {bg_style}')

    def test_transform_padding_equals_dataloader_padding(self):
        """
        Padding by the transform must be indistinguishable from the segmentation having been padded with that
        label beforehand, which is what nnU-Net's dataloader does at real image borders: rotated and scaled
        patches that reach past the crop must not cut the segmentation short.
        """
        rng = np.random.RandomState(12)
        for i in range(8):
            shape = (10, 12, 14)
            seg = torch.from_numpy(blobs(rng, shape, rng.randint(2, 5)))[None]
            grid = torch.from_numpy(rng.uniform(-1.25, 1.25, size=(6, 7, 8, 3))).float()
            for pad in (0, -1):
                for bg_style in (True, False):
                    for tiebreak in ('lowest', 'highest'):
                        t = transform((6, 7, 8), tiebreak, bg_style, border_mode_seg='constant', padding_value_seg=pad)
                        out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None)[0]
                        ref = padded_argmax(seg, grid, 'bilinear', pad, tiebreak).to(out.dtype)
                        self.assertTrue(torch.equal(out, ref), f'{int((out != ref).sum())} voxels (i={i}, {pad}, '
                                                               f'{bg_style}, {tiebreak})')

    def test_constant_image_padding_is_padding(self):
        """
        'constant' must be the image padded with padding_value_image and resampled, like 'zeros' with 0: checked
        against an image that is padded for real, with sample points inside, across and beyond the border.
        """
        rng = np.random.RandomState(6)
        margin = 8
        for shape in ((6, 11), (5, 9, 14)):
            img = torch.from_numpy(rng.rand(2, *shape)).float()
            grid = torch.from_numpy(rng.uniform(-1.3, 1.3, size=(7, 8, 9)[:len(shape)] + (len(shape),))).float()
            g = grid.clone()
            for k, n in enumerate(shape[::-1]):  # grid axis k is spatial axis dim - 1 - k
                g[..., k] = (2 * (((g[..., k] + 1) * n - 1) / 2 + margin) + 1) / (n + 2 * margin) - 1
            for value in (0., -1., 5.):
                padded = torch.nn.functional.pad(img, [margin] * (2 * len(shape)), value=value)
                for mode in ('bilinear', 'nearest') + (('bicubic',) if len(shape) == 2 else ()):
                    t = SpatialTransform(patch_size=grid.shape[:-1], patch_center_dist_from_border=0, random_crop=False,
                                         mode_image=mode, padding_mode_image='constant', padding_value_image=value)
                    out = t._apply_to_image(img, grid=grid, center_location_in_pixels=None)
                    ref = grid_sample(padded[None], g[None], mode=mode, padding_mode='border', align_corners=False)[0]
                    self.assertTrue(torch.allclose(out, ref, atol=1e-5), f'{shape} {value} {mode}: '
                                                                          f'{float((out - ref).abs().max())}')
            # 'zeros' must not have moved at all
            t = SpatialTransform(patch_size=grid.shape[:-1], patch_center_dist_from_border=0, random_crop=False)
            self.assertTrue(torch.equal(t._apply_to_image(img, grid=grid, center_location_in_pixels=None),
                                        grid_sample(img[None], grid[None], mode='bilinear', padding_mode='zeros',
                                                    align_corners=False)[0]))

    def test_constant_padding_keeps_the_image(self):
        """
        With nnU-Net's border_mode_seg='constant' and a non-zero padding value, sampling a non-cubic
        segmentation exactly at its own pixel centres must return it unchanged - no padding value inside.
        """
        rng = np.random.RandomState(7)
        for shape in ((6, 20), (4, 9, 24)):
            seg = torch.from_numpy(blobs(rng, shape, 3))[None]
            axes = [torch.tensor([grid_coord(x, n) for x in range(n)]) for n in shape]
            grid = torch.stack(torch.meshgrid(*axes, indexing='ij')[::-1], dim=-1).float()
            t = transform(shape, 'nearest', False, border_mode_seg='constant', padding_value_seg=-1)
            out = t._apply_to_segmentation(seg, grid=grid, center_location_in_pixels=None)
            self.assertTrue(torch.equal(out, seg), f'{int((out != seg).sum())} voxels changed ({shape})')

    def test_unknown_tiebreak_is_rejected(self):
        with self.assertRaises(ValueError):
            SpatialTransform(patch_size=(4, 4), patch_center_dist_from_border=0, random_crop=False,
                             seg_tiebreak='bogus')


if __name__ == '__main__':
    unittest.main()
