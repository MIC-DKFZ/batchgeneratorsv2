import abc
import warnings

import torch


class HedgingMahKeysYo(dict):
    def __setitem__(self, key, value):
        if key in self.keys():
            warnings.warn(f'Key {key} is already present in parameter dictionary. Will be overwritten')
        super().__setitem__(key, value)


class BasicTransform(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

        if data_dict.get('mask') is not None:
            data_dict['mask'] = self._apply_to_mask(data_dict['mask'], **params)

        if data_dict.get('keypoints') is not None:
            data_dict['keypoints'] = self._apply_to_keypoints(data_dict['keypoints'], **params)

        if data_dict.get('bbox') is not None:
            data_dict['bbox'] = self._apply_to_bbox(data_dict['bbox'], **params)

        return data_dict

    def _apply_to_image(self, img, **params):
        pass

    def _apply_to_mask(self, mask, **params):
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        pass

    def _apply_to_bbox(self, bbox, **params):
        pass

    def get_parameters(self, **data_dict) -> HedgingMahKeysYo:
        return HedgingMahKeysYo()


class RandomTransform(BasicTransform):
    def __init__(self, p_per_sample: float = 1):
        super().__init__()
        self.p_per_sample = p_per_sample

    def get_parameters(self, **data_dict) -> dict:
        dct = super(RandomTransform, self).compute_parameters(**data_dict)
        dct['apply_to_sample'] = torch.rand(1).item() < self.p_per_sample
        return dct

    def apply(self, data_dict, **params):
        if params['apply_to_sample']:
            return super().apply(data_dict, **params)
        else:
            return data_dict


class ExpensiveParamsTransform(RandomTransform):
    def get_parameters(self, **data_dict) -> dict:
        dct = super().get_parameters(**data_dict)
        # very expensive shit
        sleep(60)
        return dct

class ImageOnlyTransform(RandomTransform):
    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

        return data_dict


class PPerColorChannelTransform(ImageOnlyTransform):



if __name__ == '__main__':
    pass