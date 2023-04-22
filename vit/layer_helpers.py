# Copied from https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/layer_helpers.py
# as allowed by the MIT license at https://github.com/hila-chefer/Transformer-Explainability/blob/main/LICENSE
#
# May contain minor modifications.

""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
