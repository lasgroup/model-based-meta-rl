import torch


class AffineTransform:

    def __init__(self, normalization_mean, normalization_std):

        self.loc_tensor = torch.asarray(normalization_mean, dtype=torch.float32)
        self.scale_tensor = torch.asarray(normalization_std, dtype=torch.float32)

        if torch.numel(self.scale_tensor) == 1:
            self.scale_mat = self.scale_tensor * torch.eye(self.loc_tensor.shape[-1])
        else:
            if self.scale_tensor.ndim == 2:
                assert self.scale_tensor.shape[0] == 1
            self.scale_mat = torch.diag(self.scale_tensor.squeeze())

    def apply(self, base_dist):
        if isinstance(base_dist, torch.distributions.Categorical):
            assert torch.count_nonzero(self.loc_tensor) == 0
            assert torch.count_nonzero(self.scale_tensor - 1.0) == 0
            return base_dist
        else:
            base_dist = base_dist

            if isinstance(base_dist, torch.distributions.Normal):
                transformed_dist = torch.distributions.Normal(
                    loc=((self.scale_mat @ base_dist.loc) + self.loc_tensor),
                    scale=(base_dist.scale.log() + self.scale_tensor.log()).exp()
                )
            elif isinstance(base_dist, torch.distributions.MixtureSameFamily):
                assert isinstance(base_dist.component_distribution, torch.distributions.Normal)
                transformed_component_distribution = torch.distributions.Normal(
                    loc=(self.scale_mat @ base_dist.component_distribution.loc + self.loc_tensor.unsqueeze(-1)),
                    scale=(base_dist.component_distribution.scale.log() + self.scale_tensor.unsqueeze(-1).log()).exp()
                )
                transformed_dist = torch.distributions.MixtureSameFamily(
                    base_dist.mixture_distribution, transformed_component_distribution
                )
            else:
                raise NotImplementedError

            transformed_dist.transform = [self.scale_mat, self.loc_tensor]
            transformed_dist.base_dist = base_dist

            # def cdf(y, **kwargs):
            #     x = torch.inverse(self.scale_mat) @ (y - self.loc_tensor)
            #     return base_dist.cdf(x, **kwargs)
            #
            # def mean():
            #     return self.scale_mat @ base_dist.mean + self.loc_tensor
            #
            # def stddev():
            #     return torch.exp(torch.log(base_dist.stddev) + torch.log(self.scale_tensor))
            #
            # def variance():
            #     return torch.exp(torch.log(base_dist.variance) + 2 * torch.log(self.scale_tensor))
            #
            # transformed_dist.cdf = cdf
            # transformed_dist.mean = mean
            # transformed_dist.stddev = stddev
            # transformed_dist.variance = variance

            return transformed_dist
