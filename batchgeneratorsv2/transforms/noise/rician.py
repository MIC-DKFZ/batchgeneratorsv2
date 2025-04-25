import torch
from typing import Tuple
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class RicianNoiseTransform(ImageOnlyTransform):
    """
    Adds Rician noise to simulate MRI characteristics.

    Args:
        noise_variance (Tuple[float, float]): Range to sample Gaussian noise variance used in Rician computation.
    """

    def __init__(self, noise_variance: Tuple[float, float] = (0.0, 0.1)):
        super().__init__()
        self.noise_variance = noise_variance

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        variance = float(np.random.uniform(*self.noise_variance))
        return {'variance': variance}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        var = params['variance']
        noise_real = torch.empty_like(img).normal_(mean=0.0, std=var)
        noise_imag = torch.empty_like(img).normal_(mean=0.0, std=var)

        min_val = img.min()
        shifted = img - min_val

        rician = torch.sqrt((shifted + noise_real).pow_(2).add_(noise_imag.pow_(2)))
        rician = rician + min_val

        # Normalize to match original mean and std
        input_mean, input_std = img.mean(), img.std()
        rician_mean, rician_std = rician.mean(), rician.std()

        if rician_std > 0:
            rician = (rician - rician_mean) / rician_std * input_std + input_mean
        else:
            rician = rician * 0 + input_mean  # fallback if std is zero (flat image)

        return rician


if __name__ == '__main__':
    import torch
    import numpy as np

    # Create a synthetic normalized 3D image: (C, D, H, W)
    image = torch.ones(1, 32, 64, 64) * 0.5  # z-score normalized MRI-like noise
    image[0,1,1,1] = 2

    # Instantiate the transform
    transform = RicianNoiseTransform(noise_variance=(0.05, 0.1))

    # Sample parameters and apply transform
    params = transform.get_parameters(image=image)
    image_noisy = transform._apply_to_image(image, **params)

    from batchviewer import view_batch
    view_batch(image, image_noisy)