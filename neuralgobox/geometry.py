from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Box:
    lower: Tensor
    upper: Tensor

    def intersect(self, other: Box) -> Box:
        return Box(torch.maximum(self.lower, other.lower),
                   torch.minimum(self.upper, other.upper))

    def minkowski_sum(self, other: Box) -> Box:
        return Box(self.lower + other.lower, self.upper + other.upper)

    @property
    def center(self) -> Tensor:
        return (self.lower + self.upper) / 2

    @property
    def offset(self) -> Tensor:
        return (self.upper - self.lower) / 2

    def separation(self, other: Box) -> Tensor:
        return torch.abs(self.center - other.center) - self.offset - other.offset


@dataclass
class Transform:
    scale: Tensor
    shift: Tensor

    def __call__(self, box: Box) -> Box:
        return Box(box.lower * self.scale + self.shift,
                   box.upper * self.scale + self.shift)


@dataclass
class Role:
    range: Box
    error: Box
    transform: Transform

    def existential(self, concept_box: Box) -> Box:
        return self.transform(self.range.intersect(concept_box)).minkowski_sum(self.error)
