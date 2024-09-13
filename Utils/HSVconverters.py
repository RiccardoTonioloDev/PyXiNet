import torch


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r, g, _ = image.unbind(dim=-3)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(image, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(
        dim=-3
    )

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = (
        gc.add_(4.0)
        .sub_(rc)
        .mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))
    )

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    # vpqt -> rgb mapping based on i
    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long
    )
    select = select.to(device=img.device, non_blocking=True)

    select = select[:, i]
    if select.ndim > 3:
        # if input.shape is (B, ..., C, H, W) then
        # select.shape is (C, B, ...,  H, W)
        # thus we move C axis to get (B, ..., C, H, W)
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)
