import torch
from rllib.util.value_estimation import discount_cumsum


def lambda_values(next_values: torch.Tensor, rewards: torch.Tensor,
                  discount: float, lambda_: float) -> torch.Tensor:
    tds = rewards + (1. - lambda_) * discount * next_values
    tds[-1] = tds[-1].add(lambda_ * discount * next_values[-1])
    return discount_cumsum(tds, lambda_ * discount)


def discount_sequence(factor, length):
    d = torch.cumprod(factor * torch.ones((length,)), dim=0) / factor
    return d

