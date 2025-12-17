from typing import Tuple, Union
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class MoveChannelsTransform(BasicTransform):
    def __init__(self, channel_ids: Union[int, Tuple[int, ...]], source_key: str, target_key: str):
        super().__init__()
        if isinstance(channel_ids, int):
            channel_ids = (channel_ids,)
        self.channel_ids: Tuple[int, ...] = tuple(channel_ids)
        self.source_key = source_key
        self.target_key = target_key

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def apply(self, data_dict, **params):
        src = data_dict[self.source_key]
        assert src.ndim in (3, 4), f"Expected (C,X,Y) or (C,X,Y,Z), got {src.shape}"

        if self.target_key in data_dict:
            tgt = data_dict[self.target_key]
            assert src.ndim == tgt.ndim, "source and target key must have the same number of dimensions"
            assert src.shape[1:] == tgt.shape[1:], (
                f"spatial dimensions must match. Got source: {src.shape} and target: {tgt.shape}"
            )
        else:
            tgt = None

        C = src.shape[0]
        idx = torch.as_tensor(self.channel_ids, device=src.device, dtype=torch.long)

        keep = torch.ones(C, device=src.device, dtype=torch.bool)
        keep[idx] = False

        move = src[~keep]      # channels to move
        src_new = src[keep]    # remaining channels

        # attach moved channels to target
        if tgt is None:
            data_dict[self.target_key] = move
        else:
            data_dict[self.target_key] = torch.cat((tgt, move), dim=0)

        # update or remove source
        if src_new.shape[0] == 0:
            del data_dict[self.source_key]
        else:
            data_dict[self.source_key] = src_new

        return data_dict