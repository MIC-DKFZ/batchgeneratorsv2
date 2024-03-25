from copy import deepcopy

import torch

from batchgenerators_torch.transforms.base.basic_transform import BasicTransform


class Convert3DTo2DTransform(BasicTransform):
    def get_parameters(self, **data_dict) -> dict:
        return {}

    def apply(self, data_dict, **params):
        if 'image' in data_dict.keys():
            data_dict['original_shape_image'] = deepcopy(data_dict['image']).shape
        if 'segmentation' in data_dict.keys():
            data_dict['original_shape_segmentation'] = deepcopy(data_dict['segmentation']).shape
        if 'regression_target' in data_dict.keys():
            data_dict['original_shape_image_regression_target'] = deepcopy(data_dict['regression_target']).shape
        return super().apply(data_dict, **params)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        shp = img.shape
        return img.reshape((shp[0] * shp[1], *shp[2:]))

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_to_image(segmentation, **params)

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError


class Convert2DTo3DTransform(BasicTransform):
    def get_parameters(self, **data_dict) -> dict:
        return {i: data_dict[i] for i in
                ['original_shape_image', 'original_shape_segmentation', 'original_shape_image_regression_target']
                if i in data_dict.keys()}

    def apply(self, data_dict, **params):
        data_dict = super().apply(data_dict, **params)
        if 'original_shape_image' in data_dict.keys():
            del data_dict['original_shape_image']
        if 'original_shape_image_regression_target' in data_dict.keys():
            del data_dict['original_shape_image_regression_target']
        if 'original_shape_segmentation' in data_dict.keys():
            del data_dict['original_shape_segmentation']
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return img.reshape(params['original_shape_image'])

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return regression_target.reshape(params['original_shape_image_regression_target'])

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation.reshape(params['original_shape_segmentation'])

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError


if __name__ == '__main__':
    d = torch.rand((2, 32, 64, 128))
    s = torch.ones((1, 32, 64, 128))

    fwd = Convert3DTo2DTransform()
    bwd = Convert2DTo3DTransform()

    inp = {'image': d, 'segmentation': s}

    tmp = fwd(**inp)
    print(tmp['image'].shape, tmp['segmentation'].shape)
    out = bwd(**tmp)
    print(out['image'].shape, out['segmentation'].shape)
