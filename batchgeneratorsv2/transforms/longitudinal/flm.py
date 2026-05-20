from collections.abc import Sequence
import torch
import numpy as np
from scipy import ndimage as nd

from fake_lesion_mask import fake_lesion_mask, OPTIONS

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class FakeLesionMask(BasicTransform):

    def __init__(self,
                 params: list = OPTIONS,
                 target_class: int = 1,
                 regions_where_class_cannot_be: Sequence[int] = None
                 ):

        self.params = params
        self.target_class = target_class
        self.regions_where_class_cannot_be = regions_where_class_cannot_be
        self.structure = None

    def apply(self, data_dict, **params):
        data_dict['image'] = self._apply_to_image(data_dict['image'], data_dict['segmentation'], **params)
        return data_dict

    def get_parameters(self, **data_dict):
        if self.structure is None:
            dim = data_dict.get('image').ndim - 1 # C, H, W, (D)
            self.structure = nd.generate_binary_structure(dim, dim)
        return {'flm_params': self.params}

    def get_where_target_class_cannot_be(self, seg: np.ndarray):
        if self.regions_where_class_cannot_be is None:
            return np.array([False])
        return np.isin(seg.ravel(), self.regions_where_class_cannot_be).reshape(seg.shape)

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params):
        flm_params = params['flm_params']
        seg_as_numpy = seg.squeeze(0).numpy().copy() # segs are stored and read as 1, H, W, D
        lesion_mask = seg_as_numpy == self.target_class
        augmented_lesion_mask = fake_lesion_mask(lesion_mask, params=flm_params, structure=self.structure)
        augmented_lesion_mask &= ~self.get_where_target_class_cannot_be(seg_as_numpy)
        img[-1] = torch.from_numpy(augmented_lesion_mask).to(img.dtype)
        return img


if __name__ == "__main__":
    import nibabel as nib
    import os
    # let's mimic here what goes on during dataloading
    images = [
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/If_68.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/If_31.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/If_127.nii.gz',
    ]

    segs = [
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/Mf_68.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/Mf_31.nii.gz',
        '/Users/vicentcaselles/work/research/timelessegv2/data-gen-new-15-05-26/Mf_127.nii.gz',
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

        save(image_out[0], im, im_data.affine)
        save(image_out[1], im.replace('.nii.gz', '_baseline.nii.gz'), im_data.affine)
        save(seg_out[0], seg, im_data.affine)
