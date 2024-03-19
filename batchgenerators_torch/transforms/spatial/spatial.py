import torch

from batchgenerators_torch.transforms.base.basic_transform import BasicTransform


class SpatialTransform(BasicTransform):
    def __init__(self):
        super().__init__()

    def get_parameters(self, **data_dict) -> dict:
        pass

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError
