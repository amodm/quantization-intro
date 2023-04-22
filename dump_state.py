#!/usr/bin/env python3
import operator
import sys
import torch
from collections import OrderedDict
from functools import reduce


def dump_model(model, spaces = ''):
    params = 0
    for k,v in model.items():
        if isinstance(v, torch.Tensor):
            params += reduce(operator.mul, v.size())
            shape = " x ".join(map(lambda x: str(x), v.size()))
            v_min = torch.min(v)
            v_max = torch.max(v)
            ts = f'tensor[{ shape }]'
            print(f'{spaces}{k:<32} {ts:<28} min={v_min:+.8f} max={v_max:.8f}')
        else:
            print(f'{spaces}{k:<32} non-tensor {type(v)}')
            if isinstance(v, dict) or isinstance(data, OrderedDict):
                params + dump_model(v, f'{spaces}  ')
    if spaces == '':
        print(f'# total parameters = {human_count(params, base=1000, suffix="")}; @f32 = {human_count(params * 4)}; @f16 = {human_count(params * 2)}; @f8 = {human_count(params)}')
    return params


def human_count(n, base=1024, suffix='B'):
    if n < base:
        return f'{n} {suffix}'
    elif n < base**2:
        return f'{int(n/base)} K{suffix}'
    elif n < base**3:
        return f'{int(n/base**2)} M{suffix}'
    else:
        return f'{int(n/base**3)} G{suffix}'


if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} <model.pth>')
    exit(1)

model_file = sys.argv[1]
data = torch.load(model_file)
dump_model(data)

