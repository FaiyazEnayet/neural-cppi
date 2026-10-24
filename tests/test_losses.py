"""Stage 3 P0-A: anytime-shortfall loss sign and component tests.

The round-2 multi-horizon objective subtracted the anytime (pathwise)
floor-shortfall term from a minimization objective, rewarding the very
shortfall it was meant to penalize. These tests pin the corrected loss
in ``src/dhl/experiments/losses.py``:

- a positive anytime shortfall raises the minimization loss;
- a one-step finite-difference gradient matches autograd;
- zero anytime weight reproduces the impA-form loss exactly (the
  anytime path is ignored, so impA-era numbers stay comparable);
- all four named components are reported and gradients stay finite.
"""

import torch

from dhl.experiments.cppi_learned import FLOOR, FLOOR_PENALTY
from dhl.experiments.losses import multihorizon_loss


def _paths(dip):
    """Two paths with the same terminal wealth; one dips below the floor."""
    vt = torch.tensor([100.0, 100.0], dtype=torch.float64)
    path = torch.tensor(
        [[100.0, 100.0 - dip, 100.0], [100.0, 100.0, 100.0]],
        dtype=torch.float64,
    )
    return vt, path


def test_stage3_neural_cppi_0_anytime_shortfall_raises_loss():
    vt, path_flat = _paths(0.0)
    _, path_dip = _paths(5.0)  # dips to 95, below the 100 * 0.9 floor zone
    floor = 96.0
    total_flat, comp_flat = multihorizon_loss(
        vt, path_flat, floor=floor, anytime_penalty=15.0
    )
    total_dip, comp_dip = multihorizon_loss(
        vt, path_dip, floor=floor, anytime_penalty=15.0
    )
    assert comp_flat["anytime_shortfall"] == 0.0
    assert comp_dip["anytime_shortfall"] > 0.0
    # The corrected sign: more anytime shortfall -> higher loss.
    assert total_dip > total_flat
    # And the increase is exactly the weighted component.
    assert torch.isclose(
        total_dip - total_flat,
        15.0 * comp_dip["anytime_shortfall"],
        atol=1e-9,
    )


def test_stage3_neural_cppi_0_finite_difference_matches_autograd():
    vt = torch.tensor([95.0, 103.0], dtype=torch.float64,
                      requires_grad=True)
    path = torch.tensor([[100.0, 92.0, 95.0], [100.0, 101.0, 103.0]],
                        dtype=torch.float64, requires_grad=True)
    total, _ = multihorizon_loss(vt, path, floor=100.0,
                                 anytime_penalty=15.0)
    total.backward()

    eps = 1e-6

    def total_at(vt_v, path_v):
        t, _ = multihorizon_loss(
            torch.tensor(vt_v, dtype=torch.float64),
            torch.tensor(path_v, dtype=torch.float64),
            floor=100.0, anytime_penalty=15.0,
        )
        return float(t)

    fd_vt0 = (total_at([95.0 + eps, 103.0], path.detach().tolist())
              - total_at([95.0 - eps, 103.0], path.detach().tolist())
              ) / (2 * eps)
    assert abs(fd_vt0 - float(vt.grad[0])) < 1e-6
    fd_path = (total_at(vt.detach().tolist(),
                        [[100.0, 92.0 + eps, 95.0], [100.0, 101.0, 103.0]])
               - total_at(vt.detach().tolist(),
                          [[100.0, 92.0 - eps, 95.0], [100.0, 101.0, 103.0]])
               ) / (2 * eps)
    assert abs(fd_path - float(path.grad[0, 1])) < 1e-6


def test_stage3_neural_cppi_0_zero_anytime_weight_reproduces_impa_loss():
    vt, path = _paths(40.0)  # deep dip; must be ignored at zero weight
    crash_w = torch.tensor([3.0, 1.0], dtype=torch.float64)
    total, comp = multihorizon_loss(
        vt, path, crash_weights=crash_w, floor=FLOOR, anytime_penalty=0.0
    )
    terminal_sf = (crash_w * torch.relu(FLOOR - vt)).mean()
    expected = -vt.mean() + FLOOR_PENALTY * terminal_sf
    assert torch.isclose(total, expected, atol=1e-9)
    assert comp["anytime_shortfall"] > 0.0  # measured, just not weighted


def test_stage3_neural_cppi_0_components_reported_and_grads_finite():
    vt = torch.tensor([80.0, 120.0], dtype=torch.float64,
                      requires_grad=True)
    path = torch.tensor([[100.0, 85.0, 80.0], [100.0, 110.0, 120.0]],
                        dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.0, 0.5, 0.9], [0.0, 0.2, 0.2]],
                           dtype=torch.float64)
    total, comp = multihorizon_loss(
        vt, path, floor=100.0, anytime_penalty=15.0,
        weight_path=weights, turnover_penalty=2.0,
    )
    assert set(comp) == {
        "neg_expected_terminal_wealth",
        "terminal_shortfall",
        "anytime_shortfall",
        "turnover",
    }
    assert all(torch.isfinite(v) for v in comp.values())
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(vt.grad).all()
    assert torch.isfinite(path.grad).all()
