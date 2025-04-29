import sys
import math
import numpy as np
import torch

from utils.configuration import config


def sw(prediction_model_3d, batch_test_image_3d, device):
    _, _, *sdims = batch_test_image_3d.shape
    sdimst = tuple(sdims)

    win_size = config["win_size"]

    srng = []

    for idx, x in zip(range(3), config["intersection"]):
        srng.append(int(win_size[idx] * (1 - x)))

    scnt = []
    for x in range(3):
        cnt = int(math.ceil(float(sdimst[x]) / srng[x]))
        for x in (d for d in range(cnt) if d * srng[x] + win_size[x] >= sdimst[x]):
            m = x
            break
        scnt.append(m + 1 if m is not None else 1)

    beg = []
    for dim in range(3):
        dim_beg = []
        for idx in range(scnt[dim]):
            begi = idx * srng[dim]
            begi -= max(begi + win_size[dim] - sdimst[dim], 0)
            dim_beg.append(begi)
        beg.append(dim_beg)
    prod = np.asarray([x.flatten() for x in np.meshgrid(*beg, indexing="ij")]).T
    pieces = [tuple(slice(p, p + win_size[y]) for y, p in enumerate(x)) for x in prod]

    win_cnt = len(pieces)

    win_span = range(0, win_cnt)

    prodls = []
    numls = []
    cus = []

    for x in win_span:
        pi = range(x, min(x + 1, win_cnt))
        up = [[slice(idx // win_cnt, idx // win_cnt + 1), slice(None)] + list(pieces[idx % win_cnt]) for idx in pi]
        bi = batch_test_image_3d[up[0]].to(device)
        sgen = prediction_model_3d(bi)

        bx = torch.ones(win_size, device=device)
        cus = list((sgen,))

        for x in range(len(cus)):
            cs = cus[x].shape
            if len(prodls) <= x:
                pred_s = [1, cs[1]]
                prodls.append(torch.zeros(pred_s, device=device))
                numls.append(torch.zeros([1, 1] + pred_s[2:], device=device))
                by = bx
                for i in pieces:
                    numls[-1][(slice(None), slice(None), *i)] += by
            cus[x] *= bx

    for x in range(len(prodls)):
        prodls[x] /= numls.pop(0)

    batch_prediction = prodls[0]

    return batch_prediction
