import torch
import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar


class CutOffOutliersTransform(ImageOnlyTransform):
    """
    Clamps intensities in the image to percentiles to remove outliers,
    and optionally rescales the result to retain original standard deviation.

    Args:
        percentile_lower (RandomScalar): Lower cutoff percentile (0-100).
        percentile_upper (RandomScalar): Upper cutoff percentile (0-100).
        p_synchronize_channels (bool): If True, same percentiles are used for all channels.
        p_per_channel (float): Probability to apply cutoff to each channel.
        p_retain_std (float): Probability of retaining the original standard deviation after clipping.
    """

    def __init__(self,
                 percentile_lower: RandomScalar = 0.2,
                 percentile_upper: RandomScalar = 99.8,
                 p_synchronize_channels: bool = False,
                 p_per_channel: float = 1.0,
                 p_retain_std: float = 1.0):
        super().__init__()
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.p_synchronize_channels = p_synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_std = p_retain_std

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if self.p_synchronize_channels:
            lower = float(sample_scalar(self.percentile_lower))
            upper = float(sample_scalar(self.percentile_upper))
            percentiles = [(lower, upper) if apply else None for apply in apply_channel]
        else:
            percentiles = []
            for apply in apply_channel:
                if not apply:
                    percentiles.append(None)
                else:
                    lower = float(sample_scalar(self.percentile_lower))
                    upper = float(sample_scalar(self.percentile_upper))
                    percentiles.append((lower, upper))

        retain_std_flags = [
            np.random.rand() < self.p_retain_std if p is not None else False
            for p in percentiles
        ]

        return {'percentiles': percentiles, 'retain_std': retain_std_flags}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        percentiles = params['percentiles']
        retain_std = params['retain_std']

        for c, perc in enumerate(percentiles):
            if perc is None:
                continue

            img_c = img[c]
            if retain_std[c]:
                orig_std = img_c.std()

            # Use numpy only to calculate percentiles
            img_c_np = img_c.detach().cpu().numpy()
            lower_val = np.percentile(img_c_np, perc[0])
            upper_val = np.percentile(img_c_np, perc[1])

            img_c_clipped = img_c.clamp(min=float(lower_val), max=float(upper_val))

            if retain_std[c]:
                clipped_std = img_c_clipped.std()
                if clipped_std > 1e-8:
                    img_c_clipped = (img_c_clipped - img_c_clipped.mean()) / clipped_std * orig_std + img_c_clipped.mean()

            img[c] = img_c_clipped

        return img

if __name__ == '__main__':
    from batchviewer import view_batch

    image = torch.randn(1, 32, 64, 64) * 5

    transform = CutOffOutliersTransform(
        percentile_lower=(0.5, 5),
        percentile_upper=(95, 99.5),
        p_synchronize_channels=True,
        p_per_channel=1.0,
        p_retain_std=0.5
    )

    params = transform.get_parameters(image=image)
    image_clipped = transform._apply_to_image(image.clone(), **params)

    view_batch(image, image_clipped, image_clipped-image)