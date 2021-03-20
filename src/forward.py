import torch

def chop(*args, model=None, shave=12, min_size=16000000):#160000
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    # if self.input_large:
    #     scale = 1
    # else:
    #     scale = self.scale[self.idx_scale]
    scale = 1
    # n_GPUs = min(self.n_GPUs, 4)
    n_GPUs = 1
    _, _, h, w = args[0].size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    list_x = [[
        a[:, :, 0:h_size, 0:w_size],
        a[:, :, 0:h_size, (w - w_size):w],
        a[:, :, (h - h_size):h, 0:w_size],
        a[:, :, (h - h_size):h, (w - w_size):w]
    ] for a in args]

    list_y = []
    if w_size * h_size < min_size:
        for i in range(0, 4, n_GPUs):
            x = [torch.cat(_x[i:(i + n_GPUs)], dim=0) for _x in list_x]
            y = model(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.extend(_y.chunk(n_GPUs, dim=0))
    else:
        for p in zip(*list_x):
            y = chop(*p, forward_function=model, shave=shave, min_size=min_size)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    b, c, _, _ = list_y[0][0].size()
    y = [_y[0].new(b, c, h, w) for _y in list_y]
    for _list_y, _y in zip(list_y, y):
        _y[:, :, :h_half, :w_half] \
            = _list_y[0][:, :, :h_half, :w_half]
        _y[:, :, :h_half, w_half:] \
            = _list_y[1][:, :, :h_half, (w_size - w + w_half):]
        _y[:, :, h_half:, :w_half] \
            = _list_y[2][:, :, (h_size - h + h_half):, :w_half]
        _y[:, :, h_half:, w_half:] \
            = _list_y[3][:, :, (h_size - h + h_half):, (w_size - w + w_half):]

    if len(y) == 1: y = y[0]

    return y

def x8(*args, model=None, pow=1, squeeze=True):
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 0:
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 1:
            tfnp = v2np[:, :, ::-1, :].copy()
        else:
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(v.data.device)
        return ret

    list_x = []
    for a in args:
        x = [a]
        for tf in 0, 1, 2: x.extend([_transform(_x, tf) for _x in x])

        list_x.append(x)

    list_y = []
    for x in zip(*list_x):
        y = model(*x)
        if not isinstance(y, list): y = [y]
        if not list_y:
            list_y = [[_y] for _y in y]
        else:
            for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    for _list_y in list_y:
        for i in range(len(_list_y)):
            if i > 3:
                _list_y[i] = _transform(_list_y[i], 2)
            if i % 4 > 1:
                _list_y[i] = _transform(_list_y[i], 1)
            if (i % 4) % 2 == 1:
                _list_y[i] = _transform(_list_y[i], 0)

    if pow == 1:
        y = [torch.cat(_y, dim=0).mean(dim=0) for _y in list_y]
    else:
        def cat_y(_y):
            res = torch.cat(_y, dim=0)
            mean = res.mean(dim=0, keepdim=True)
            bias = res - mean
            bias_sgn = bias.sgn()
            bias = bias.abs()
            mean_bias = (bias_sgn * (bias + 1e-16).pow(pow)).mean(dim=0)
            res = res.mean(dim=0) + mean_bias.sgn() * mean_bias.abs().pow(1 / pow)
            return res
        y = [cat_y(_y) for _y in list_y]
    if len(y) == 1 and squeeze: y = y[0]

    return y

def x8crop4(*args, model=None, pow=1, squeeze=True):
    l = len(args)
    argsx4 = [[] for i in range(4)]
    for x in args:
        w, h = x.shape[-2:]
        ww, hh = (w // 32) * 16, (h // 32) * 16
        argsx4[0].append(x[..., :ww, :hh])
        argsx4[1].append(x[..., ww:, :hh])
        argsx4[2].append(x[..., :ww, hh:])
        argsx4[3].append(x[..., ww:, hh:])
    ys = [x8(*a, model=model, pow=pow, squeeze=False) for a in argsx4]
    res = []
    for i in range(l):
        y1, y2, y3, y4 = ys[0][i], ys[1][i], ys[2][i], ys[3][i]
        y12 = torch.cat([y1, y2], dim=-2)
        y34 = torch.cat([y3, y4], dim=-2)
        y = torch.cat([y12, y34], dim=-1)
        res.append(y)
    if len(res) == 1 and squeeze: res = res[0]
    return res

def x8crop9(*args, model=None, pow=1, squeeze=True, padding=128, method=x8):
    """
    0 4 1
    5 8 6
    2 7 3
    """
    l = len(args)
    argsx9 = [[] for i in range(9)]
    params = []
    for x in args:
        w, h = x.shape[-2:]
        ww, hh = (w // 32) * 16, (h // 32) * 16
        x9, y9 = (w // 64) * 16, (h // 64) * 16
        argsx9[0].append(x[..., :ww, :hh])
        argsx9[1].append(x[..., ww:, :hh])
        argsx9[2].append(x[..., :ww, hh:])
        argsx9[3].append(x[..., ww:, hh:])
        argsx9[4].append(x[..., x9:x9 + ww, :hh])
        argsx9[5].append(x[..., :ww, y9:y9 + hh])
        argsx9[6].append(x[..., ww:, y9:y9 + hh])
        argsx9[7].append(x[..., x9:x9 + ww, hh:])
        argsx9[8].append(x[..., x9:x9 + ww, y9:y9 + hh])
        shape = x.shape[:-2]
        if shape[0] == 1 and len(shape) > 1:
            shape = shape[1:]
        params.append((w, h, ww, hh, x9, y9, x.device, shape))

    ys = [method(*a, model=model, pow=pow, squeeze=False) for a in argsx9]
    res = []
    for i in range(l):
        y0, y1, y2, y3 = ys[0][i], ys[1][i], ys[2][i], ys[3][i]
        y4, y5, y6, y7 = ys[4][i], ys[5][i], ys[6][i], ys[7][i]
        y8 = ys[8][i]
        w, h, ww, hh, x9, y9, device, shape = params[i]
        wstep = 1 / (ww - x9 - (padding // 2))
        wmask = torch.cat((
            torch.zeros(x9),
            torch.arange(0, 1 + wstep, wstep)[:ww - x9 - (padding // 2)],
            torch.ones(padding),
            torch.arange(1, -wstep, -wstep)[:x9 - (padding - (padding // 2))],
            torch.zeros(w - (x9 + ww)),
        )).unsqueeze(1).to(device)
        hstep = 1 / (hh - y9 - (padding // 2))
        hmask = torch.cat((
            torch.zeros(y9),
            torch.arange(0, 1 + hstep, hstep)[:hh - y9 - (padding // 2)],
            torch.ones(padding),
            torch.arange(1, -hstep, -hstep)[:y9 - (padding - (padding // 2))],
            torch.zeros(h - (y9 + hh)),
        )).unsqueeze(0).to(device)
        basemask = torch.stack([
            torch.cat([wmask] * h, dim=1),
            torch.cat([hmask] * w, dim=0),
        ], dim=0)
        mask = torch.max(basemask, dim=0).values
        cmask = torch.min(basemask, dim=0).values
        y01 = torch.cat([y0, y1], dim=-2)
        y23 = torch.cat([y2, y3], dim=-2)
        yy = torch.cat([y01, y23], dim=-1)
        wy = torch.cat([y4, y7], dim=-1)
        hy = torch.cat([y5, y6], dim=-2)
        wy = torch.cat([torch.zeros((*shape, x9, h)).to(device), wy, torch.zeros((*shape, w - (x9 + ww), h)).to(device)], dim=-2)
        hy = torch.cat([torch.zeros((*shape, w, y9)).to(device), hy, torch.zeros((*shape, w, h - (y9 + hh))).to(device)], dim=-1)
        cy = torch.cat([torch.zeros((*shape, x9, hh)).to(device), y8, torch.zeros((*shape, w - (x9 + ww), hh)).to(device)], dim=-2)
        cy = torch.cat([torch.zeros((*shape, w, y9)).to(device), cy, torch.zeros((*shape, w, h - (y9 + hh))).to(device)], dim=-1)
        wy = cy * hmask + wy * (1 - hmask)
        hy = cy * wmask + hy * (1 - wmask)
        why = wy * (mask - hmask + cmask / 2) + hy * (mask - wmask + cmask / 2)
        #wy = wy * wmask + yy * (1 - wmask)
        #hy = hy * hmask + yy * (1 - hmask)
        y = yy * (1 - mask) + why#(wy + hy) / 2
        res.append(y)
    if len(res) == 1 and squeeze: res = res[0]
    return res
