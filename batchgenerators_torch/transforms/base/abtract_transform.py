from enum import Enum

import numpy as np
import abc
from typing import List, Union

import torch


class ATYPE(Enum):
    IMAGE = 0
    SEGMENTATION = 1
    POINT = 2





class Convert3DTo2DTransform(BasicTransform):
    supported_input_types = [ATYPE.IMAGE, ATYPE.SEGMENTATION]

    def __init__(self, apply_to, input_types):
        super().__init__(apply_to, input_types, p_per_sample=1, p_per_channel=1, synchronize_channels=True, synchronize_keys=True)
        assert self.supported_input_types is not None, f"{self.__class__.__name__} " \
                                                         f"needs to define __supported_input_types"

    def transform_sample(self, *args):
        output = []
        for i in args:
            shp = i[0].shape
            output.append(i[0].reshape((shp[0] * shp[1], shp[2], shp[3])))
        return output


class BasicTransform(abc.ABC):
    # needs to be defined by child classes
    supported_input_types = None

    def __init__(self,
                 apply_to: Union[str, List[str]],
                 input_types: Union[ATYPE, List[ATYPE]],
                 p_per_sample: float = 1, p_per_channel: float = 1,
                 synchronize_keys: bool = True, synchronize_channels: bool = False):
        """
        :param apply_to: Key or list of keys the transform should be applied to
        :param input_types: str or list of strings (same length as apply_to) that describes what input types the keys represent (for example image, segmentation, point)
        :param p_per_sample: probability of this transform for being applied to each sample in the batch
        :param p_per_channel: probability of this transform for being applied to each channel in the sample
        :param synchronize_keys: whether the same transformation (same seed) should be applied to all keys in apply_to
        :param synchronize_channels: whether the same transformation should be applied to all selected
                (determined by p_per_channel) channels of a sample (interacts with synchronize_keys).
        """
        self.apply_to = apply_to if isinstance(apply_to, (list, tuple)) else [apply_to]
        self.input_types = input_types if isinstance(input_types, (list, tuple)) else [input_types]
        assert len(self.apply_to) == len(self.input_types)
        self.key_atypes = {i: j for i, j in zip(self.apply_to, self.input_types)}
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.synchronize_keys = synchronize_keys
        self.synchronize_channels = synchronize_channels
        assert all([i in self.supported_input_types for i in self.input_types]), \
            f"Unsupported input types. Got {input_types}, supported: {self.supported_input_types}"

    def __call__(self, **kwargs):
        if self.synchronize_keys:
            result = self._apply_to_batch(**{i: kwargs[i] for i in self.apply_to})
        else:
            result = {}
            for k in self.apply_to:
                result[k] = self._apply_to_batch(**{k: kwargs[k]})
        kwargs.update(result)
        return kwargs

    def _apply_to_batch(self, **kwargs):
        keys = list(kwargs.keys())
        num_channels = kwargs[keys[0]].shape[1]
        num_samples = kwargs[keys[0]].shape[0]
        ret_batch = {k: [] for k in keys}
        for b in range(num_samples):
            ret_sample = {k: [] for k in keys}
            if np.random.uniform() < self.p_per_sample:
                if self.synchronize_channels:
                    assert all([num_channels == kwargs[k].shape[1] for k in keys[1:]])
                    channel_mask = np.random.uniform(size=num_channels) < self.p_per_channel
                    ret = self.transform_sample(*[(kwargs[k][b][channel_mask], self.key_atypes[k]) for k in keys])
                    for ch in range(num_channels):
                        # decide if we pick the original or from ret
                        if channel_mask[ch]:
                            for i, k in enumerate(keys):
                                idx = sum(channel_mask[:ch])
                                import IPython;IPython.embed()
                                ret_sample[k].append(ret[i][None, idx:idx + 1])
                        else:
                            for k in keys:
                                ret_sample[k].append(kwargs[k][b:b+1][ch:ch+1])
                else:
                    for c in range(num_channels):
                        if np.random.uniform() < self.p_per_channel:
                            ret = self.transform_sample(*[(kwargs[k][b][c:c+1], self.key_atypes[k]) for k in keys])
                            for i, k in enumerate(keys):
                                ret_sample[k].append(ret[i][None])
                        else:
                            for k in keys:
                                ret_sample[k].append(kwargs[k][b:b+1][c:c+1])
            else:
                for k in keys:
                    ret_sample[k].append(kwargs[k][b:b+1])
            for k in keys:
                ret_batch[k].append(torch.stack(ret_sample[k], 0))
        return ret_batch

    @abc.abstractmethod
    def transform_sample(self, *args):
        """
        args is a tuple of inputs that need to be processed with _the same_ transformation.

        """


if __name__ == '__main__':
    a = Convert3DTo2DTransform('data', ATYPE.IMAGE)
    b = a(**{'data': torch.rand((2, 3, 128, 128, 128))})
    print(b['data'].shape)
