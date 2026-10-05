import torch as th

from neuralgobox.geometry import Box
from neuralgobox.models import EMBED_DIM as D, NeuralGOBoxModel

N_IPR, NG, NZ, NR = 16, 40, 60, 5


def make():
    th.manual_seed(0)
    return NeuralGOBoxModel(N_IPR, NG, NZ, NR, 'cpu')


def toy_normal_forms():
    th.manual_seed(1)
    return (th.randint(0, NG + NZ, (50, 2)),
            th.randint(0, NG + NZ, (20, 3)),
            th.randint(0, NG + NZ, (5, 2)),
            th.stack([th.randint(0, NR, (20,)), th.randint(0, NG + NZ, (20,)),
                      th.randint(0, NG + NZ, (20,))], 1),
            th.stack([th.randint(0, NG + NZ, (20,)), th.randint(0, NR, (20,)),
                      th.randint(0, NG + NZ, (20,))], 1))


def box(center, offset):
    return Box(lower=center - offset, upper=center + offset)


def test_forward_backward():
    net = make()
    logits = net(th.randn(7, N_IPR))
    assert logits.shape == (7, NG), f'bad logits shape {logits.shape}'
    (logits.sum() + net.el_loss(toy_normal_forms())).backward()
    assert all(p.grad is not None for n, p in net.named_parameters()
               if p.requires_grad and not n.startswith('class_center_bn'))
    print('ok  forward/backward')


def test_frozen_parameters():
    net = make()
    for p in (net.relation_error_center.weight, net.relation_error_raw_offset.weight,
              net.hf_shift):
        assert not p.requires_grad and p.abs().max().item() == 0.0
    print('ok  error boxes and hf_shift are zero and frozen')


def test_scorer_reads_unrescaled_parameters():
    net = make()
    net.eval()
    x = th.randn(7, N_IPR)
    a = net(x)
    net._frame = net.frame()
    b = net(x)
    net._frame = None
    assert th.equal(a, b), 'rescaling leaked into the scorer'
    print('ok  scorer is unaffected by the rescaling')


def test_rescaling_preserves_axioms():
    net = make()
    ids = th.arange(NG)
    raw = net.get_class_box(ids, normalize=False)
    net._frame = net.frame()
    scaled = net.get_class_box(ids)
    net._frame = None
    d_raw = raw.separation(Box(raw.lower[:1], raw.upper[:1]))
    d_scaled = scaled.separation(Box(scaled.lower[:1], scaled.upper[:1]))
    assert th.equal(d_raw > 0, d_scaled > 0)
    print('ok  rescaling preserves the sign of every separation')


def test_disjoint_zero_set_is_sound():
    net = make()
    a = box(th.zeros(1, D), th.ones(1, D) * 0.1)
    b = box(th.ones(1, D) * 5.0, th.ones(1, D) * 0.1)
    assert net.disjoint_loss(a, b).item() == 0.0
    inter = a.intersect(b)
    assert (inter.lower > inter.upper).any().item()
    print('ok  disjoint_loss zero set implies disjointness')


def test_inclusion_zero_set_is_sound():
    net = make()
    inner = box(th.zeros(1, D), th.ones(1, D) * 0.1)
    outer = box(th.zeros(1, D), th.ones(1, D) * 10.0)
    assert net.inclusion_loss(inner, outer).item() == 0.0
    assert (outer.lower <= inner.lower).all() and (inner.upper <= outer.upper).all()
    assert net.inclusion_loss(outer, inner).item() > 0.0
    print('ok  inclusion_loss zero set implies containment')


def test_gci2_overlap_term_present():
    net = make()
    gci2 = toy_normal_forms()[4]
    c, d = net.get_class_box(gci2[:, 0]), net.get_class_box(gci2[:, 2])
    r = net.get_role(gci2[:, 1])
    assert (net.gci2_loss(gci2) >= net.inclusion_loss(c, r.existential(d)) - 1e-6).all()
    print('ok  gci2 includes the range/filler overlap term')



def test_predict_zero_agrees_with_forward_on_seen_ids():
    net = make()
    net.eval()
    x = th.randn(7, N_IPR)
    assert th.allclose(net(x), net.predict_zero(x, net.all_gos), atol=1e-6)
    print('ok  predict_zero on the training vocabulary reproduces forward')


def test_predict_zero_scores_zero_classes():
    net = make()
    net.eval()
    x = th.randn(7, N_IPR)
    out = net.predict_zero(x, th.arange(NG, NG + NZ))
    assert out.shape == (7, NZ), f'bad zero logits shape {out.shape}'
    out.sum().backward()
    grad = net.class_center.weight.grad
    assert grad[NG:].abs().sum() > 0, 'no gradient reached the zero classes'
    assert grad[:NG].abs().sum() == 0, 'seen classes moved on a zero-only score'
    print('ok  predict_zero reaches zero classes and only those')


def test_el_loss_reaches_zero_classes():
    net = make()
    net.train()
    net.el_loss(toy_normal_forms()).backward()
    assert net.class_center.weight.grad[NG:].abs().sum() > 0
    print('ok  el_loss delivers gradient to zero classes')


if __name__ == '__main__':
    test_forward_backward()
    test_frozen_parameters()
    test_scorer_reads_unrescaled_parameters()
    test_rescaling_preserves_axioms()
    test_disjoint_zero_set_is_sound()
    test_inclusion_zero_set_is_sound()
    test_gci2_overlap_term_present()
    test_predict_zero_agrees_with_forward_on_seen_ids()
    test_predict_zero_scores_zero_classes()
    test_el_loss_reaches_zero_classes()
    print('\nall passed')
