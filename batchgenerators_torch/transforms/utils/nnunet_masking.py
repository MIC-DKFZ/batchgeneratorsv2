from typing import List

from batchgenerators_torch.transforms.base.basic_transform import BasicTransform


class MaskImageTransform(BasicTransform):
    def __init__(self,
                 apply_to_channels: List[int],
                 channel_idx_in_seg: int = 0,
                 set_outside_to: float = 0,
                 replace_mask_in_seg_with: int = None
                 ):
        super().__init__()
        self.apply_to_channels = apply_to_channels
        self.channel_idx_in_seg = channel_idx_in_seg
        self.set_outside_to = set_outside_to
        self.replace_mask_in_seg_with = replace_mask_in_seg_with

    def apply(self, data_dict, **params):
        seg = data_dict.get('segmentation')
        mask = seg[self.channel_idx_in_seg] < 0
        image = data_dict.get('image')
        for c in range(image.shape[0]):
            image[c][mask] = self.set_outside_to

        if self.replace_mask_in_seg_with is not None:
            seg[self.channel_idx_in_seg] = self.replace_mask_in_seg_with
            data_dict['segmentation'] = seg
        data_dict['image'] = image
        return data_dict

    def get_parameters(self, **data_dict) -> dict:
        return {}

