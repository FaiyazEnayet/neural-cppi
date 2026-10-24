"""Multi-horizon training losses with explicit named components.

The loss is a minimization objective built from four separately
reported components:

- ``neg_expected_terminal_wealth``: -E[V_T]
- ``terminal_shortfall``: E[crash_w * relu(floor - V_T)], weighted by
  ``FLOOR_PENALTY`` (the repo convention in ``cppi_learned``)
- ``anytime_shortfall``: E[crash_w * relu(floor - min_t V_t)], the
  pathwise floor shortfall, weighted by ``anytime_penalty``
- ``turnover``: mean absolute change of the risky weight along the
  path, weighted by ``turnover_penalty``

Every component ADDS to the loss. An earlier private round-2 objective
subtracted the anytime term, so a larger pathwise shortfall lowered
the loss and the two candidates trained with a positive anytime weight
were corrupted tests of the hypothesis; the corrected sign here is
pinned by ``tests/test_losses.py``. With ``anytime_penalty=0`` the
total reproduces the original terminal-only objective exactly, so
zero-weight runs stay comparable with earlier results.
"""

from __future__ import annotations

import torch

from .cppi_learned import FLOOR, FLOOR_PENALTY

COMPONENTS = (
    "neg_expected_terminal_wealth",
    "terminal_shortfall",
    "anytime_shortfall",
    "turnover",
)


def multihorizon_loss(
    v_terminal: torch.Tensor,
    path: torch.Tensor | None = None,
    *,
    crash_weights: torch.Tensor | None = None,
    weight_path: torch.Tensor | None = None,
    floor: float = FLOOR,
    floor_penalty: float = FLOOR_PENALTY,
    anytime_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``(total, components)`` for one training batch.

    ``v_terminal`` is the per-path terminal wealth, shape ``(batch,)``;
    ``path`` is the full wealth path, shape ``(batch, T)`` (required
    when ``anytime_penalty`` may apply; the anytime component is still
    measured and reported when the weight is zero). ``crash_weights``
    are the per-path crash multipliers used by the repo's convention
    (ones when omitted). ``weight_path`` is the per-step risky weight,
    shape ``(batch, T)``, used only for the turnover component.
    """
    if crash_weights is None:
        crash_weights = torch.ones_like(v_terminal)

    neg_wealth = -v_terminal.mean()
    terminal_shortfall = (
        crash_weights * torch.relu(floor - v_terminal)
    ).mean()

    if path is not None:
        path_min = path.min(dim=1).values
        anytime_shortfall = (
            crash_weights * torch.relu(floor - path_min)
        ).mean()
    else:
        anytime_shortfall = torch.zeros_like(neg_wealth)

    if weight_path is not None and weight_path.shape[1] > 1:
        turnover = weight_path.diff(dim=1).abs().mean()
    else:
        turnover = torch.zeros_like(neg_wealth)

    components = {
        "neg_expected_terminal_wealth": neg_wealth,
        "terminal_shortfall": terminal_shortfall,
        "anytime_shortfall": anytime_shortfall,
        "turnover": turnover,
    }
    total = (
        neg_wealth
        + floor_penalty * terminal_shortfall
        + anytime_penalty * anytime_shortfall
        + turnover_penalty * turnover
    )
    return total, components
