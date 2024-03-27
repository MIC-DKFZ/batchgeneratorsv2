from typing import List

from batchgenerators_torch.transforms.base.basic_transform import BasicTransform


class ComposeTransforms(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def apply(self, data_dict, **params):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict
