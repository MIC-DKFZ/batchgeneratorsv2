import numpy as np
import torch
from typing import Union, Tuple, List, Callable
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class ColorFunctionExtractor:
    def __init__(self, rectangle_value: Union[int, float, Tuple[float, float], Callable]):
        self.rectangle_value = rectangle_value

    def __call__(self, x: torch.Tensor) -> float:
        if np.isscalar(self.rectangle_value):
            return float(self.rectangle_value)
        elif callable(self.rectangle_value):
            return float(self.rectangle_value(x))
        elif isinstance(self.rectangle_value, (tuple, list)):
            return float(np.random.uniform(*self.rectangle_value))
        else:
            raise RuntimeError("Unrecognized format for rectangle_value")


class BlankRectangleTransform(ImageOnlyTransform):
    """
    Overwrites random rectangles in the image with a constant or sampled value.

    Supports 2D/3D data and various configurations of rectangle size/value.
    """

    def __init__(self,
                 rectangle_size: Union[int,
                                       Tuple[int, ...],
                                       Tuple[Tuple[int, int], ...]],
                 rectangle_value: Union[int, float, Tuple[float, float], Callable],
                 num_rectangles: Union[int, Tuple[int, int]],
                 force_square: bool = False,
                 p_per_channel: float = 1.0):
        super().__init__()
        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles
        self.force_square = force_square
        self.p_per_channel = p_per_channel
        self.color_fn = ColorFunctionExtractor(rectangle_value)

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        spatial_shape = image.shape[1:]
        D = len(spatial_shape)

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        # Number of rectangles
        if isinstance(self.num_rectangles, int):
            n_rects = [self.num_rectangles for _ in range(C)]
        else:
            n_rects = [np.random.randint(self.num_rectangles[0], self.num_rectangles[1]) for _ in range(C)]

        # Precompute all rectangles for all channels
        rectangles = [[] for _ in range(C)]
        for c in range(C):
            if not apply_channel[c]:
                continue
            for _ in range(n_rects[c]):
                size = self._sample_rectangle_size(D)
                lb = [np.random.randint(0, spatial_shape[d] - size[d] + 1) for d in range(D)]
                ub = [lb[d] + size[d] for d in range(D)]
                rectangles[c].append((lb, ub))

        return {
            'apply_channel': apply_channel,
            'rectangles': rectangles
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        out = img.clone()
        for c, (apply, rects) in enumerate(zip(params['apply_channel'], params['rectangles'])):
            if not apply:
                continue
            for lb, ub in rects:
                slices = tuple([slice(l, u) for l, u in zip(lb, ub)])
                intensity = self.color_fn(out[c][slices])
                out[c][slices] = intensity
        return out

    def _sample_rectangle_size(self, ndim: int) -> List[int]:
        if isinstance(self.rectangle_size, int):
            return [self.rectangle_size] * ndim

        elif isinstance(self.rectangle_size, (tuple, list)) and all(isinstance(x, int) for x in self.rectangle_size):
            return list(self.rectangle_size)

        elif isinstance(self.rectangle_size, (tuple, list)) and all(isinstance(x, (tuple, list)) for x in self.rectangle_size):
            if self.force_square:
                val = np.random.randint(*self.rectangle_size[0])
                return [val] * ndim
            else:
                return [np.random.randint(*self.rectangle_size[d]) for d in range(ndim)]

        raise RuntimeError("Unrecognized format for rectangle_size")


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from skimage.data import camera
    # from skimage.util import img_as_float32
    #
    # img = torch.from_numpy(img_as_float32(camera())).unsqueeze(0)  # (C, H, W)
    #
    # transform = BlankRectangleTransform(
    #     rectangle_size=((10, 30), (20, 40)),
    #     rectangle_value=(0.0, 1.0),
    #     num_rectangles=(2, 5),
    #     force_square=False,
    #     p_per_channel=1.0
    # )
    #
    # params = transform.get_parameters(image=img)
    # img_aug = transform._apply_to_image(img, **params)
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(img.squeeze().numpy(), cmap='gray')
    # plt.title("Original")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_aug.squeeze().numpy(), cmap='gray')
    # plt.title("With Blank Rectangles")
    #
    # plt.tight_layout()
    # plt.show()
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a random 3D image (C, D, H, W)
    image = torch.rand(1, 32, 64, 64)  # Single-channel 3D volume

    # Instantiate the transform
    transform = BlankRectangleTransform(
        rectangle_size=((4, 10), (10, 20), (10, 20)),  # (Z, Y, X) size ranges
        rectangle_value=(0.0, 1.0),  # Random intensity per rectangle
        num_rectangles=(3, 7),  # 3 to 6 rectangles per channel
        force_square=False,
        p_per_channel=1.0  # Always apply to the channel
    )

    # Sample transform parameters and apply
    params = transform.get_parameters(image=image)
    image_aug = transform._apply_to_image(image, **params)

    from batchviewer import view_batch
    view_batch(image, image_aug)