import unittest
import warnings
from itertools import product
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from batchgeneratorsv2.helpers.fft_conv import fft_conv, irfftn_spatial

# The references deliberately avoid torch.fft.irfftn on CPU: on Intel CPUs it can corrupt the heap for some of these
# shapes (see irfftn_spatial). numpy's FFT is pocketfft, not oneMKL. In (5, 96, 40) and (3, 64, 33) the second spatial
# dim is more than 16x as long as the first one, the case oneMKL mishandles.
SPATIAL_SHAPES = [(30,), (31,), (12, 20), (12, 21), (5, 96, 40), (3, 64, 33), (4, 6, 8, 10)]


class TestIrfftnSpatial(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(1234)

    def test_matches_numpy_and_round_trips(self):
        for spatial, batch in product(SPATIAL_SHAPES, ((1, 1), (2, 3))):
            signal = torch.randn(*batch, *spatial, generator=self.generator)
            dims = tuple(range(2, signal.ndim))
            spectrum = torch.fft.rfftn(signal, dim=dims)
            with self.subTest(spatial=spatial, batch=batch):
                out = irfftn_spatial(spectrum, spatial)
                expected = np.fft.irfftn(spectrum.numpy(), s=spatial, axes=dims)
                self.assertEqual(out.shape, signal.shape)
                self.assertEqual(out.dtype, torch.float32)
                np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-5)
                torch.testing.assert_close(out, signal, rtol=0, atol=1e-5)

    def test_cpu_complex_fft_gets_the_batch_dims_outermost(self):
        # the complex FFT must transform the innermost dims of a contiguous tensor, never the layout with the
        # half-spectrum dim as innermost batch dim
        calls = []
        real_ifftn = torch.fft.ifftn

        def recording_ifftn(x, *args, dim=None, **kwargs):
            calls.append((x.is_contiguous(), tuple(dim), x.ndim))
            return real_ifftn(x, *args, dim=dim, **kwargs)

        spectrum = torch.fft.rfftn(torch.randn(1, 1, 5, 96, 40, generator=self.generator), dim=(2, 3, 4))
        with mock.patch('torch.fft.ifftn', recording_ifftn):
            irfftn_spatial(spectrum, (5, 96, 40))
        self.assertEqual(calls, [(True, (3, 4), 5)])


class TestFFTConv(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(1234)

    def test_matches_direct_convolution(self):
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
        # (signal shape, kernel shape, groups)
        cases = [((2, 3, 50), (4, 3, 5), 1), ((1, 4, 17, 23), (4, 2, 3, 5), 2), ((1, 2, 17, 22), (3, 2, 4, 3), 1),
                 ((1, 1, 5, 96, 40), (1, 1, 3, 1, 1), 1), ((1, 1, 5, 96, 40), (1, 1, 1, 7, 1), 1),
                 ((2, 1, 6, 20, 21), (1, 1, 3, 5, 3), 1)]
        for (signal_shape, kernel_shape, groups), padding in product(cases, (0, 'same')):
            signal = torch.randn(signal_shape, generator=self.generator)
            kernel = torch.randn(kernel_shape, generator=self.generator)
            with self.subTest(signal=signal_shape, kernel=kernel_shape, groups=groups, padding=padding):
                out = fft_conv(signal, kernel, padding=padding, groups=groups)
                with warnings.catch_warnings():  # torch's note on padding='same' with even kernel sizes (the 4)
                    warnings.filterwarnings('ignore', message="Using padding='same' with even kernel lengths")
                    expected = conv[signal.ndim - 2](signal, kernel, padding=padding, groups=groups)
                self.assertEqual(out.shape, expected.shape)
                torch.testing.assert_close(out, expected, rtol=0, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
