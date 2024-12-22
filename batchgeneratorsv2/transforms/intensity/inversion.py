import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class InvertImageTransform(ImageOnlyTransform):
    def __init__(self, p_invert_image: float, p_synchronize_channels: float = 1, p_per_channel: float = 1):
        super().__init__()
        self.p_invert_image = p_invert_image
        self.p_synchronize_channels = p_synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply = np.random.uniform() < self.p_invert_image
        if apply:
            if np.random.uniform() < self.p_synchronize_channels:
                apply_to_channel = torch.arange(0, shape[0])
            else:
                apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        else:
            apply_to_channel = []
        return {
            'apply_to_channel': apply_to_channel,
            'apply': apply,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply']:
            return img
        else:
            for ch in params['apply_to_channel']:
                mn = img[ch].mean()
                img[ch] -= mn
                img[ch] *= -1
                img[ch] += mn
        return img


if __name__ == '__main__':
    mbt = InvertImageTransform(0.5, 0.5, 0.5)
    from batchviewer import view_batch

    for _ in range(100):
        data_dict = {'image': torch.ones((2, 20, 192, 64))}
        data_dict['image'][0, :10] = -1
        data_dict['image'][1, :5] = -1
        ret = mbt(**data_dict)
        print(ret['image'][0, 0, 0, 0], ret['image'][1, 0, 0, 0])
    view_batch(mbt(**data_dict)['image'])