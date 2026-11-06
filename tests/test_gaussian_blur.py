import unittest
from itertools import product

import numpy as np
import torch
from scipy.ndimage import correlate1d
from torch.nn.functional import pad

from batchgeneratorsv2.transforms.noise.gaussian_blur import (
    DEFAULT_TRUNCATE, GaussianBlurTransform, _FFT_TRANSPOSE_MIN_STRIDE, _build_kernel, _compute_kernel_size,
    _next_fast_fft_len, _reflect_pad_any_width, blur_dimension)


def _mirror_reference(x: torch.Tensor, sigma: float, axis: int) -> np.ndarray:
    """float64 correlation with the same kernel; scipy's 'mirror' is torch's 'reflect' (d c b | a b c d | c b a)"""
    kernel = _build_kernel(sigma, truncate=DEFAULT_TRUNCATE).double().numpy()
    return correlate1d(x.double().numpy(), kernel, axis=axis, mode='mirror')


class TestBlurDimension(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(1234)

    def test_fft_matches_direct_and_float64_reference(self):
        # the shapes cover the strided FFT (short stride) and the transposed one (stride >= _FFT_TRANSPOSE_MIN_STRIDE)
        shapes = [(2, 75), (1, 40, 33), (3, 30, 600), (1, 600, 20), (2, 12, 20, 18), (1, 9, 31, 700), (1, 40, 30, 20)]
        self.assertTrue(any(np.prod(s[d + 2:]) >= _FFT_TRANSPOSE_MIN_STRIDE for s in shapes for d in range(len(s) - 1)))
        for shape, sigma in product(shapes, (0.5, 1.0, 2.5, 6.0)):
            x = torch.randn(shape, generator=self.generator)
            for dim in range(len(shape) - 1):
                with self.subTest(shape=shape, sigma=sigma, dim=dim):
                    fft = blur_dimension(x, sigma, dim, force_use_fft=True)
                    direct = blur_dimension(x, sigma, dim, force_use_fft=False)
                    self.assertEqual(fft.shape, x.shape)
                    self.assertEqual(fft.dtype, x.dtype)
                    self.assertTrue(fft.is_contiguous())
                    torch.testing.assert_close(fft, direct, rtol=0, atol=1e-5)
                    np.testing.assert_allclose(fft.double().numpy(), _mirror_reference(x, sigma, dim + 1), rtol=0, atol=1e-5)

    def test_input_is_not_modified(self):
        for force_use_fft in (False, True):
            x = torch.randn((2, 10, 600), generator=self.generator)
            x_copy = x.clone()
            for dim in range(2):
                blur_dimension(x, 3.0, dim, force_use_fft=force_use_fft)
            self.assertTrue(torch.equal(x, x_copy))

    def test_kernel_wider_than_axis(self):
        # torch's reflect padding needs pad < axis length; wider kernels continue the reflection periodically
        for length, sigma, force_use_fft in product((2, 3, 5, 16), (2.0, 5.25, 10.0, 30.0), (False, True)):
            x = torch.randn((2, length, 7), generator=self.generator)
            with self.subTest(length=length, sigma=sigma, force_use_fft=force_use_fft):
                out = blur_dimension(x, sigma, 0, force_use_fft=force_use_fft)
                np.testing.assert_allclose(out.double().numpy(), _mirror_reference(x, sigma, 1), rtol=0, atol=1e-5)

    def test_axis_of_length_one_is_repeated(self):
        x = torch.randn((3, 1, 9), generator=self.generator)
        for force_use_fft in (False, True):
            torch.testing.assert_close(blur_dimension(x, 2.0, 0, force_use_fft=force_use_fft), x, rtol=0, atol=1e-6)

    def test_reflect_pad_any_width_matches_torch(self):
        for length, width in product(range(2, 10), range(0, 9)):
            if width >= length:
                continue
            x = torch.randn((2, 3, length, 4), generator=self.generator)
            with self.subTest(length=length, width=width):
                self.assertTrue(torch.equal(_reflect_pad_any_width(x, width, 2),
                                            pad(x, [0, 0, width, width, 0, 0], mode='reflect')))

    def test_direct_is_multiply_then_add_in_kernel_order(self):
        # the documented arithmetic, written with numpy float32 ufuncs: bit-identical on every machine. Includes shapes
        # and channel counts for which torch's own convolution would pick different implementations.
        shapes = [(1, 300), (3, 64), (1, 64, 48), (2, 31, 40), (1, 20, 36, 30), (3, 12, 20, 18), (1, 40, 64, 56)]
        for shape, sigma in product(shapes, (0.5, 0.75, 1.0, 2.5, 5.0)):
            x = torch.randn(shape, generator=self.generator)
            for dim in range(len(shape) - 1):
                k = _build_kernel(sigma, truncate=DEFAULT_TRUNCATE)
                xp = _reflect_pad_any_width(x, k.shape[0] // 2, dim + 1).numpy()
                kn, n = k.numpy(), shape[dim + 1]
                expected = np.take(xp, np.arange(0, n), axis=dim + 1) * kn[0]
                for t in range(1, len(kn)):
                    expected = expected + np.take(xp, np.arange(t, t + n), axis=dim + 1) * kn[t]
                with self.subTest(shape=shape, sigma=sigma, dim=dim):
                    out = blur_dimension(x, sigma, dim, force_use_fft=False)
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertTrue(out.is_contiguous())
                    np.testing.assert_array_equal(out.numpy(), expected)

    def test_direct_is_independent_of_thread_count(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        x = torch.randn((2, 40, 70, 52), generator=self.generator)
        outs = {}
        for threads in (1, 3):
            torch.set_num_threads(threads)
            outs[threads] = [blur_dimension(x, s, d, force_use_fft=False) for s in (0.5, 1.0, 3.0) for d in range(3)]
        for a, b in zip(outs[1], outs[3]):
            self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available(), 'needs a GPU')
    def test_direct_on_gpu_equals_cpu(self):
        for shape in ((2, 30, 41), (1, 20, 33, 27)):
            x = torch.randn(shape, generator=self.generator)
            for dim, sigma in product(range(len(shape) - 1), (0.5, 1.0, 4.0)):
                with self.subTest(shape=shape, dim=dim, sigma=sigma):
                    out = blur_dimension(x.cuda(), sigma, dim, force_use_fft=False)
                    self.assertTrue(out.is_cuda)
                    self.assertTrue(torch.equal(out.cpu(), blur_dimension(x, sigma, dim, force_use_fft=False)))

    def test_next_fast_fft_len(self):
        self.assertEqual(_next_fast_fft_len(518), 540)
        self.assertEqual(_next_fast_fft_len(512), 512)
        self.assertEqual(_next_fast_fft_len(7), 8)
        self.assertEqual(_next_fast_fft_len(1), 1)


class TestGaussianBlurTransform(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def test_benchmark_key_is_the_size_of_the_kernel_used(self):
        for sigma in (0.5, 0.6, 1.0, 2.0):
            with self.subTest(sigma=sigma):
                t = GaussianBlurTransform(sigma, benchmark=True)
                t.benchmark_num_runs = 1
                t(image=torch.randn(1, 12, 14, 13))
                keys = {k for per_shape in t.benchmark_use_fft.values() for k in per_shape}
                self.assertEqual(keys, {_build_kernel(sigma, truncate=DEFAULT_TRUNCATE).shape[0]})
                self.assertEqual(keys, {_compute_kernel_size(sigma, truncate=DEFAULT_TRUNCATE)})

    def test_no_channel_drawn_leaves_image_unchanged(self):
        for synchronize_channels, synchronize_axes, benchmark in product((False, True), repeat=3):
            with self.subTest(synchronize_channels=synchronize_channels, synchronize_axes=synchronize_axes,
                              benchmark=benchmark):
                t = GaussianBlurTransform(1., synchronize_channels, synchronize_axes, 0., benchmark=benchmark)
                x = torch.randn(2, 10, 12, 11)
                self.assertTrue(torch.equal(t(image=x.clone())['image'], x))
                self.assertEqual(t.benchmark_use_fft, {})

    @unittest.skipUnless(torch.cuda.is_available(), 'needs a GPU')
    def test_transform_on_gpu_equals_cpu(self):
        x = torch.randn(3, 20, 26, 22)
        for synchronize_channels, synchronize_axes in product((False, True), repeat=2):
            outs = []
            for device in ('cpu', 'cuda'):
                np.random.seed(0); torch.manual_seed(0)
                t = GaussianBlurTransform((0.5, 3.), synchronize_channels, synchronize_axes, 0.7, benchmark=False)
                outs.append(t(image=x.clone().to(device))['image'].cpu())
            with self.subTest(synchronize_channels=synchronize_channels, synchronize_axes=synchronize_axes):
                self.assertFalse(torch.equal(outs[0], x))
                self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_callable_sigma_gets_shape_and_dim_in_every_branch(self):
        shape = (2, 10, 12, 11)
        for synchronize_channels, synchronize_axes in product((False, True), repeat=2):
            calls = []

            def sigma(*args, **kwargs):
                calls.append((args, kwargs))
                return 1.0

            GaussianBlurTransform(sigma, synchronize_channels, synchronize_axes, 1.)(image=torch.randn(shape))
            with self.subTest(synchronize_channels=synchronize_channels, synchronize_axes=synchronize_axes):
                self.assertTrue(calls)
                for args, kwargs in calls:
                    self.assertEqual(args, (torch.Size(shape),))
                    self.assertEqual(set(kwargs), {'dim'})
                    self.assertIn(kwargs['dim'], (None,) if synchronize_axes else (1, 2, 3))


if __name__ == '__main__':
    unittest.main()
