import sys
import os
import torch
import numpy as np
from copy import deepcopy
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform, BrightnessAdditiveTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.inversion import InvertImageTransform
from batchgeneratorsv2.transforms.intensity.random_clip import CutOffOutliersTransform
from batchgeneratorsv2.transforms.local.brightness_gradient import BrightnessGradientAdditiveTransform
from batchgeneratorsv2.transforms.local.local_contrast import LocalContrastTransform
from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
from batchgeneratorsv2.transforms.local.local_smoothing import LocalSmoothingTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.noise.blank_rectangle import BlankRectangleTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
from batchgeneratorsv2.transforms.noise.rician import RicianNoiseTransform
from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.transpose import TransposeAxesTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

MASTER_SEED = 7

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def compare_outputs(output1: dict, output2: dict) -> bool:
    if output1.keys() != output2.keys():
        print(f"    - ❌ FAIL: Output dictionaries have different keys.")
        return False
    for key in output1.keys():
        val1, val2 = output1[key], output2[key]
        if isinstance(val1, torch.Tensor):
            if not torch.equal(val1, val2):
                diff = torch.abs(val1.float() - val2.float()).max()
                print(f"    - ❌ FAIL: Tensor mismatch for key '{key}'. Max difference: {diff.item()}")
                return False
    return True

def run_test_on_data(transform_class, kwargs, input_data, dimension_str):
    print(f"\n🧪 Testing {transform_class.__name__} ({dimension_str})...")
    try:
        # --- RUN 1 ---
        # Seed both numpy and torch to establish a baseline
        seed_everything(MASTER_SEED)
        transform_run1 = transform_class(**kwargs)
        output1 = transform_run1(**deepcopy(input_data))

        # --- RUN 2 ---
        # Re-seed NumPy. This simulates the real-world scenario
        # where torch's RNG state is not controlled in worker processes. If a transform
        # uses torch.rand, it will now fail this test.
        np.random.seed(MASTER_SEED)

        transform_run2 = transform_class(**kwargs)
        output2 = transform_run2(**deepcopy(input_data))

        # --- VERIFICATION ---
        if compare_outputs(output1, output2):
            print(f"    - ✅ [PASS] Outputs are identical.")
            return True
        else:
            return False
    except Exception as e:
        print(f"    - ❌ [ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("--- Starting Determinism Test Pipeline ---")
    
    # --- A single, comprehensive list of all transforms and their base kwargs ---
    all_transforms_and_kwargs = [
        (MultiplicativeBrightnessTransform, {'multiplier_range': (0.8, 1.2), 'synchronize_channels': False}),
        (BrightnessAdditiveTransform, {'mu': 0.0, 'sigma': 0.1, 'per_channel': False}), 
        (ContrastTransform, {'contrast_range': BGContrast((0.9, 1.1)), 'preserve_range': True, 'synchronize_channels': False}),
        (GammaTransform, {'gamma': (0.8, 1.2), 'p_invert_image': 0.0, 'synchronize_channels': False, 'p_per_channel': 1, 'p_retain_stats': 0.5}),
        (GaussianNoiseTransform, {'noise_variance': (0, 0.05)}),
        (InvertImageTransform, {'p_invert_image': 0.1, 'p_synchronize_channels': 0.5, 'p_per_channel': 0.8}), 
        (CutOffOutliersTransform, {'percentile_lower': (0.1, 1.0), 'percentile_upper': (99.0, 99.9)}), 

        (BrightnessGradientAdditiveTransform, {'scale': (40, 80), 'max_strength': (0.1, 0.3)}),
        (LocalContrastTransform, {'scale': (40, 80), 'new_contrast': (0.8, 1.2)}),
        (LocalGammaTransform, {'scale': (40, 80), 'gamma': (0.8, 1.2)}),
        (LocalSmoothingTransform, {'scale': (40, 80), 'kernel_size': (0.5, 1.5)}),

        (ApplyRandomBinaryOperatorTransform, {'channel_idx': [0], 'strel_size': (1, 2)}), 
        (RemoveRandomConnectedComponentFromOneHotEncodingTransform, {'channel_idx': [0], 'fill_with_other_class_p': 0.5}),

        (BlankRectangleTransform, {'rectangle_size': ((2, 5), (2, 5), (2, 5)), 'rectangle_value': (0, 1), 'num_rectangles': (1, 2)}), 
        (GaussianBlurTransform, {'blur_sigma': (0.5, 1.5), 'benchmark': False}),
        (MedianFilterTransform, {'filter_size': 3}),
        (RicianNoiseTransform, {'noise_variance': (0, 0.05)}), 
        (SharpeningTransform, {'strength': (0.1, 0.2)}),

        (SimulateLowResolutionTransform, {'scale': (0.8, 1.0), 'synchronize_channels': False, 'synchronize_axes': False, 'ignore_axes': None}), 
        (MirrorTransform, {'allowed_axes': (0, 1, 2)}), 
        (Rot90Transform, {'num_axis_combinations': 1, 'allowed_axes': {0, 1, 2}}), 
        (SpatialTransform, {'patch_size': (12, 12, 12), 'patch_center_dist_from_border': [6, 6, 6], 'random_crop': True, 'p_elastic_deform': 0.2, 'p_rotation': 0.2, 'p_scaling': 0.2}), 
        (TransposeAxesTransform, {'allowed_axes': {0, 1, 2}}), 
    ]

    passed_count = 0
    failed_count = 0

    # ===============================================================
    # Part 1: Test all transforms on 3D data
    # ===============================================================
    print("\n--- Part 1: Testing all transforms on 3D data ---")
    seed_everything(MASTER_SEED)
    input_data_3d = {'image': torch.randn(2, 16, 16, 16)}
    for transform_class, kwargs in all_transforms_and_kwargs:
        if run_test_on_data(transform_class, kwargs, input_data_3d, "3D"):
            passed_count += 1
        else:
            failed_count += 1

    # ===============================================================
    # Part 2: Test all transforms on 2D data
    # ===============================================================
    print("\n\n--- Part 2: Testing all transforms on 2D data ---")
    seed_everything(MASTER_SEED)
    input_data_2d = {'image': torch.randn(3, 64, 64)}
    for transform_class, kwargs_base in all_transforms_and_kwargs:

        # Skip transforms that are  3D only
        if transform_class == SpatialTransform:
            print(f"\n🧪 Skipping {transform_class.__name__} (3D-only)...")
            continue
        
        # Adapt kwargs for 2D compatibility
        if 'allowed_axes' in kwargs_base: kwargs_base['allowed_axes'] = {ax for ax in kwargs_base['allowed_axes'] if ax < 2}
        if 'rectangle_size' in kwargs_base: kwargs_base['rectangle_size'] = kwargs_base['rectangle_size'][:2]

        if run_test_on_data(transform_class, kwargs_base, input_data_2d, "2D"):
            passed_count += 1
        else:
            failed_count += 1

    # ===============================================================
    # Part 3: Run composed pipeline on sample_image.jpg using ONLY changed transforms
    # ===============================================================
    print("\n\n--- Part 3: Testing composed pipeline on sample_image.jpg (fixed transforms only) ---")
    try:
        with Image.open("sample_image.jpg") as img:
            # Resize to be square to ensure all spatial transforms are compatible
            img = img.resize((512, 512))
            input_tensor_2d = TF.to_tensor(img.convert("RGB"))
        TF.to_pil_image(input_tensor_2d).save("sample_image_original.png")
        print("✅ Loaded, resized, and saved 'sample_image.jpg' as 'sample_image_original.png'.")

        # Define the set of transform classes whose internal logic we fixed
        changed_transforms = {
            ContrastTransform,
            GammaTransform,
            GaussianNoiseTransform,
            InvertImageTransform,
            MultiplicativeBrightnessTransform,
            ApplyRandomBinaryOperatorTransform,
            RemoveRandomConnectedComponentFromOneHotEncodingTransform,
            GaussianBlurTransform,
            RicianNoiseTransform,
            SimulateLowResolutionTransform,
            MirrorTransform
        }

        # Filter the main list to get only the transforms we changed
        transforms_to_compose = []
        for cls, kw_base in all_transforms_and_kwargs:
            if cls in changed_transforms and cls != SpatialTransform:
                kw = deepcopy(kw_base)
                # Adapt kwargs for 2D on the fly
                if 'allowed_axes' in kw: kw['allowed_axes'] = {ax for ax in kw['allowed_axes'] if ax < 2}
                if 'rectangle_size' in kw: kw['rectangle_size'] = kw['rectangle_size'][:2]
                transforms_to_compose.append(cls(**kw))
        
        # Wrap all selected transforms in RandomTransform with 100% probability, better for comparison with original batchgeneratorsv2 code
        composed_2d_transforms = ComposeTransforms([
            RandomTransform(t, 1.0) for t in transforms_to_compose
        ])
        
        print(f"✅ Created a composed pipeline with {len(transforms_to_compose)} fixed transforms.")

        seed_everything(MASTER_SEED)
        final_output = composed_2d_transforms(**{'image': input_tensor_2d})
        TF.to_pil_image(final_output['image']).save("sample_image_augmented.png")
        print("✅ Saved 'sample_image_augmented.png'.")


    except FileNotFoundError:
        print("❌ WARNING: 'sample_image.jpg' not found. Skipping Part 3.")
    except Exception as e:
        print(f"    - ❌ [ERROR] An exception occurred during the composed 2D test: {e}")
        import traceback
        traceback.print_exc()
        failed_count += 1

    # --- Final Summary ---
    print("\n--- Test Pipeline Finished ---")
    print(f"Total Checks: {passed_count + failed_count} | ✅ Passed: {passed_count} | ❌ Failed: {failed_count}")
    print("--------------------------------")
    
    return failed_count == 0

if __name__ == "__main__":
    if main():
        print("\n🎉 All augmentation tests passed and are deterministic!")
    else:
        print("\n🔥 Some augmentations failed the determinism check. Please review the logs above.")
        sys.exit(1)