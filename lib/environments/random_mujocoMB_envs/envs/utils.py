import torch

# https://github.com/deepmind/dm_control/blob/main/dm_control/utils/rewards.py#L93
_DEFAULT_VALUE_AT_MARGIN = 0.1


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, torch.tensor(1.0), gaussian(d, value_at_margin))

    return value


def gaussian(x, value_at_1):
    scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1).double()))
    return torch.exp(-0.5 * (x * scale) ** 2)
