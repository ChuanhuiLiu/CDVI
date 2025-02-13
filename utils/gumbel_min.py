import torch
import numpy as np
from numbers import Number
import math
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant
__all__ = ["Gumbel"]

class Gumbelmin(TransformedDistribution):
    """
    Customized Gumbel_min Distribution from torch.distribution.Gumbel
    Note original torch.distributions.gumbel.Gumbel() is a Gumbel_max distribution.
    Same usage as any Torch.distribution with overrided probability function for numeric stability modifcation

    Usage Examples:
        >>> dst = Gumbelmin(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> dst.sample()  # sample from Gumbel distribution with loc=1, scale=2
    Args:
        loc (Tensor): Location parameter of the distribution
        scale (Tensor):  Scale parameter of the distribution
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, Number) and isinstance(scale, Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps, validate_args=validate_args)
        else:
            base_dist = Uniform(
                torch.full_like(self.loc, finfo.tiny),
                torch.full_like(self.loc, 1 - finfo.eps),
                validate_args=validate_args,
            )
        transforms = [
            AffineTransform(loc=torch.ones_like(self.scale), scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=loc, scale=self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbelmin, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)


    # Explicitly defining the log probability function for Gumbel_min due to precision issues
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value -self.loc) / self.scale
        return (y - y.exp()) - self.scale.log()
    
    # Explicitly defining the cumulative function for Gumbel_min due to precision issues 
    # e.g. dist.cdf(torch.tensor([-750])) or dist.cdf(torch.tensor([-750])) raise bound error
    def cdf(self,value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value -self.loc) / self.scale
        return torch.ones_like(self.loc)- (-1*y.exp()).exp()

    # Explicitly defining the log complimentary cumulative function for Gumbel_min with clamp function to avoid inf
    # survival function clamped at 1e-50 (log-s clamped at -50)
    def logsf(self,value,eps = 15): 
        if self._validate_args:
            self._validate_sample(value)
        y = (value -self.loc) / self.scale
        y = y.clamp(max=eps)
        return -1*y.exp()
    
    
    @property
    def mean(self):
        return -1*(self.loc + self.scale * euler_constant)

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)
    
    @property
    def entropy(self):
        return self.scale.log() + (1 + euler_constant)