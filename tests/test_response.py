import pytest

torch = pytest.importorskip("torch")

from erd.response import L_dir, L_mag, erd_sequence_loss, response_delta


def test_zero_response_is_finite_and_not_dropped():
    zero = torch.zeros(32)
    one = torch.ones(32)
    assert float(L_dir(zero.unsqueeze(0), one.unsqueeze(0))) == pytest.approx(1.0)
    assert torch.isfinite(L_mag(zero.unsqueeze(0), one.unsqueeze(0))).all()


def test_centered_response_and_objective():
    original = torch.tensor([1.0, 3.0, 2.0])
    edited = torch.tensor([2.0, 1.0, 3.0])
    delta = response_delta(edited, original)
    assert float(delta.sum()) == pytest.approx(0.0)
    objective = erd_sequence_loss(delta, delta, torch.tensor(2.0))
    assert float(objective["total"]) == pytest.approx(2.0, abs=1e-6)
