from collections.abc import Sequence, Callable
import torch
import numpy as np
from scipy import ndimage as nd

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

def lesion_vol(lesion_mask: np.ndarray, spacing) -> float:
    return np.prod(spacing) * np.count_nonzero(lesion_mask)

def stable_lesion(lesion_mask: np.ndarray) -> np.ndarray:
    return lesion_mask

def as_new_lesion(lesion_mask: np.ndarray) -> np.ndarray:
    return np.zeros_like(lesion_mask, dtype=lesion_mask.dtype)

def one_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=1)

def two_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=2)

def three_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=3)

def one_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=1)

def two_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=2)

def three_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=3)

def exhaust_list(l: list, **params) -> Callable[[np.ndarray], np.ndarray]:
    if all(isinstance(ll, list) for ll in l):
        for ll in l:
            res = exhaust_list(ll, **params)
            if res is not None:
                return res

    f, ff = l
    dothis = f(**params)
    if dothis and not isinstance(ff, list):
        return ff
    elif dothis:
        return exhaust_list(ff, **params)

OPTIONS = [
    stable_lesion,
    as_new_lesion,
    one_ero,
    two_ero,
    three_ero,
    one_dil,
    two_dil,
    three_dil
]

def _fake_lesion_mask(labeled_lesions_array: np.ndarray, num_lesions: int, spacing, params: list):
    out = np.zeros_like(labeled_lesions_array, dtype=bool)
    for l in range(1, num_lesions + 1):
        # print(f'Processing lesion {l}')

        unif_draw = np.random.uniform()
        this_lesion = labeled_lesions_array == l
        lesion_volume = lesion_vol(this_lesion, spacing)

        op = np.random.choice(params)

        out += op(this_lesion)
        # print(f'vol: {lesion_volume}, unif: {unif_draw} - {op.__name__}')

    return out

def fake_lesion_mask(lesion_mask: np.ndarray, spacing, params: list, structure) -> np.ndarray:
    labeled_mask, num_lesions = nd.label(lesion_mask, structure=structure)
    return _fake_lesion_mask(labeled_mask, num_lesions, spacing, params)

FRENCH_PARAMS = [
    [lambda u, v: u < 1/8, stable_lesion],
    [lambda u, v: u < 2/8, as_new_lesion],
    [lambda u, v: u < 3/8, one_ero],
    [lambda u, v: u < 4/8, two_ero],
    [lambda u, v: u < 5/8, three_ero],
    [lambda u, v: u < 6/8, one_dil],
    [lambda u, v: u < 7/8, two_dil],
    [lambda u, v: u < 1., three_dil]
]

class FakeLesionMask(BasicTransform):
    def __init__(self,
                 params: list = OPTIONS,
                 target_class: int = 1,
                 regions_where_class_cannot_be: Sequence[int] = None
                 ):
        self.params = params
        self.target_class = target_class
        self.regions_where_class_cannot_be = regions_where_class_cannot_be
        self.structure = nd.generate_binary_structure(3, 3)

    def apply(self, data_dict, **params):
        data_dict['image'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def get_parameters(self, **data_dict):
        return {'flm_params': self.params}

    def get_where_target_class_cannot_be(self, seg: np.ndarray):
        if self.regions_where_class_cannot_be is None:
            return np.array([False])
        return np.isin(seg.ravel(), self.regions_where_class_cannot_be).reshape(seg.shape)

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params):
        flm_params = params['flm_params']
        spatial_dim = img.ndim - 1 # channels and batch
        spacing = np.array([1.] * spatial_dim)
        structure = nd.generate_binary_structure(spatial_dim, spatial_dim)
        seg_as_numpy = seg.numpy()[0] # segs are stored and read as 1, H, W, D
        lesion_mask = seg_as_numpy == self.target_class
        augmented_lesion_mask = fake_lesion_mask(lesion_mask, spacing, params=flm_params, structure=structure)
        augmented_lesion_mask &= ~self.get_where_target_class_cannot_be(seg_as_numpy)
        img[-1] = torch.from_numpy(augmented_lesion_mask).to(img.dtype)
        return img


if __name__ == "__main__":
    import nibabel as nib
    import os
    # let's mimic here what goes on during dataloading
    images = [
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/If_03217.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/If_01417.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/If_02661.nii.gz'
    ]

    segs = [
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/Mf_03217.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/Mf_01417.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new/Mf_02661.nii.gz'
    ]

    w_c_c_b = [0, 11, 13, 14, 15, 16]
    flm = FakeLesionMask(regions_where_class_cannot_be=w_c_c_b)
    
    def save(array, inpath, affine):
        opath = 'test_' + os.path.basename(inpath)
        nib.save(nib.Nifti1Image(array, affine), opath)

    for i in range(3):
        im, seg = images[i], segs[i]
        im_data = nib.load(im)
        im_array = im_data.get_fdata()
        seg_data = np.asanyarray(nib.load(seg).dataobj)

        im_array = torch.from_numpy(im_array).to(torch.float32)
        seg_data = torch.from_numpy(seg_data).to(torch.int16)

        im_array = torch.stack([
            im_array,
            torch.zeros_like(im_array)
        ])
        seg_data = seg_data.unsqueeze(0)

        data_dict = flm(**{'image': im_array, 'segmentation': seg_data})
        image_out = data_dict['image'].numpy()
        seg_out = data_dict['segmentation'].numpy()

        save(image_out[0][0], im, im_data.affine)
        save(image_out[0][1], im.replace('.nii.gz', '_baseline.nii.gz'), im_data.affine)
        save(seg_out[0], seg, im_data.affine)
