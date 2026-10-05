import math

import torch as th
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import Box, Role, Transform

EMBED_DIM = 1024
MARGIN = 0.25
BETA = 10.0
EPSILON = 1e-5


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.layer_norm(self.activation(self.linear(x))))


class NeuralGOBoxModel(nn.Module):

    def __init__(self, n_iprs, n_terms, n_zero, n_rels, device):
        super().__init__()
        self.all_gos = th.arange(n_terms).to(device)
        self._frame = None

        k = math.sqrt(1 / EMBED_DIM)
        n_classes = n_terms + n_zero
        self.class_center = self._embedding(n_classes, -k, k)
        self.class_raw_offset = self._embedding(n_classes, -k, k)
        self.relation_range_center = self._embedding(n_rels, -k, k)
        self.relation_range_raw_offset = self._embedding(n_rels, -k, k)
        # Error boxes are held at [0, 0].
        self.relation_error_center = self._embedding(n_rels, 0.0, 0.0)
        self.relation_error_raw_offset = self._embedding(n_rels, 0.0, 0.0)
        self.relation_scale_raw = self._embedding(n_rels, 1.0 - k, 1.0 + k)
        self.relation_shift = self._embedding(n_rels, -k, k)
        # Held at zero: proteins are scored against the class box itself.
        self.hf_shift = nn.Parameter(th.empty(EMBED_DIM))
        nn.init.uniform_(self.hf_shift, -k, k)
        with th.no_grad():
            self.hf_shift.zero_()
        self.hf_shift.requires_grad_(False)
        # Unused; kept so that released checkpoints load.
        self.class_center_bn = nn.BatchNorm1d(EMBED_DIM)

        self.net = nn.Sequential(
            MLPBlock(n_iprs, EMBED_DIM),
            Residual(MLPBlock(EMBED_DIM, EMBED_DIM)),
        )

    @staticmethod
    def _embedding(n, low, high):
        emb = nn.Embedding(n, EMBED_DIM)
        if low == high:
            nn.init.constant_(emb.weight, low)
            emb.weight.requires_grad_(False)
        else:
            nn.init.uniform_(emb.weight, low, high)
        return emb


    def frame(self):
        X = th.cat([self.class_center.weight,
                    self.relation_range_center.weight], dim=0)
        s = 1.0 / th.sqrt(X.var(0, unbiased=False) + EPSILON)
        return s, -s * X.mean(0)

    def get_class_box(self, ids, normalize=True) -> Box:
        center = self.class_center(ids)
        half = th.abs(self.class_raw_offset(ids)) + EPSILON
        if normalize and self._frame is not None:
            s, t = self._frame
            center, half = s * center + t, s * half
        return Box(lower=center - half, upper=center + half)

    def get_role(self, ids) -> Role:
        range_c = self.relation_range_center(ids)
        range_half = th.abs(self.relation_range_raw_offset(ids))
        error_c = self.relation_error_center(ids)
        error_half = th.abs(self.relation_error_raw_offset(ids))
        scale = th.abs(self.relation_scale_raw(ids)) + EPSILON
        shift = self.relation_shift(ids)
        if self._frame is not None:
            s, t = self._frame
            range_c, range_half = s * range_c + t, s * range_half
            error_c, error_half = s * error_c, s * error_half
            shift = s * shift + (1.0 - scale) * t
        return Role(
            range=Box(lower=range_c - range_half, upper=range_c + range_half),
            error=Box(lower=error_c - error_half, upper=error_c + error_half),
            transform=Transform(scale=scale, shift=shift),
        )


    @staticmethod
    def _reduce(y: Tensor) -> Tensor:
        b = BETA * math.log(EMBED_DIM)
        m = y.amax(dim=-1, keepdim=True).detach()
        return m.squeeze(-1) + th.log(th.exp(b * (y - m)).mean(dim=-1)) / b

    @staticmethod
    def _scale(box1: Box, box2: Box) -> Tensor:
        k = (box1.offset.abs() + box2.offset.abs()).mean(dim=-1, keepdim=True)
        return k + EPSILON

    def inclusion_loss(self, box1: Box, box2: Box) -> Tensor:
        is_empty1 = (box1.lower > box1.upper).any(dim=-1)
        v = box1.separation(box2) + 2 * box1.offset
        v = v / self._scale(box1, box2)
        loss = self._reduce(F.relu(v + MARGIN))
        return th.where(is_empty1, th.zeros_like(loss), loss)

    def overlap_loss(self, box1: Box, box2: Box) -> Tensor:
        v = box1.separation(box2) / self._scale(box1, box2)
        return self._reduce(F.relu(v + MARGIN))

    def disjoint_loss(self, box1: Box, box2: Box) -> Tensor:
        v = box1.separation(box2) / self._scale(box1, box2)
        return F.relu(MARGIN - v).mean(dim=-1)


    def gci0_loss(self, data) -> Tensor:
        return self.inclusion_loss(self.get_class_box(data[:, 0]),
                                   self.get_class_box(data[:, 1]))

    def gci1_loss(self, data) -> Tensor:
        c = self.get_class_box(data[:, 0])
        d = self.get_class_box(data[:, 1])
        e = self.get_class_box(data[:, 2])
        return self.inclusion_loss(c.intersect(d), e) + self.overlap_loss(c, d)

    def gci1_bot_loss(self, data) -> Tensor:
        return self.disjoint_loss(self.get_class_box(data[:, 0]),
                                  self.get_class_box(data[:, 1]))

    def gci3_loss(self, data) -> Tensor:
        r = self.get_role(data[:, 0])
        c = self.get_class_box(data[:, 1])
        d = self.get_class_box(data[:, 2])
        return self.inclusion_loss(r.existential(c), d) + self.overlap_loss(r.range, c)

    def gci2_loss(self, data) -> Tensor:
        c = self.get_class_box(data[:, 0])
        r = self.get_role(data[:, 1])
        d = self.get_class_box(data[:, 2])
        return self.inclusion_loss(c, r.existential(d)) + self.overlap_loss(r.range, d)

    def el_loss(self, normal_forms) -> Tensor:
        self._frame = self.frame()
        try:
            gci0, gci1, gci1_bot, gci3, gci2 = normal_forms
            loss = self.gci0_loss(gci0).mean()
            for data, fn in ((gci1, self.gci1_loss),
                             (gci1_bot, self.gci1_bot_loss),
                             (gci3, self.gci3_loss),
                             (gci2, self.gci2_loss)):
                if len(data):
                    loss = loss + fn(data).mean()
            return loss
        finally:
            self._frame = None


    def score(self, features, go_ids):
        x = self.net(features)
        box = self.get_class_box(go_ids, normalize=False)
        o = box.offset
        w = o / o.mean(dim=-1, keepdim=True)
        return x @ (box.center * w).T + th.log(o).mean(dim=-1).view(1, -1)

    def forward(self, features):
        return self.score(features, self.all_gos)

    def predict_zero(self, features, go_ids):
        return self.score(features, go_ids)
