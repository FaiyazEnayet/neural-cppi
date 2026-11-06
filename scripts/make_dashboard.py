"""Build the interactive dashboard: docs/dashboard.html.

Trains small demo policies on the committed real data, simulates the
learned allocator against classic CPPI, and writes ONE self-contained
Plotly HTML page. No server needed: GitHub Pages can serve the file
exactly as it is.

Run:  python scripts/make_dashboard.py [--fast] [--out docs/dashboard.html]

--fast trains for 2 epochs on a single floor/horizon: enough to check
the page builds, not enough to learn anything real.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

from dhl.viz import dashboard as D

FLOORS = (85.0, 90.0, 95.0)
HORIZONS = (30, 63)
MULTIPLIERS = (3.0, 4.0, 5.0)
N_PATHS = 1000

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Results tables read by results_figures; runs/ is gitignored, so a
# fresh clone has none of these on disk.
METRICS_FILES = (
    "runs/cppi/metrics.json",
    "runs/cppi_multi/metrics.json",
    "runs/cppi_multi/drawdown_63d.json",
)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root=ROOT):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=30, check=True,
        )
        commit = out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
        return commit
    return "unknown"


def dashboard_lineage(fast=False):
    """Hashed lineage for one dashboard build.

    Records the build config, the SHA-256 of every committed
    market-data file the demo training reads, the SHA-256 of every
    runs/ metrics file embedded as a results table (or "absent" when
    the file is not on disk), and the code commit when git can
    report one. No checkpoint hash exists for the demo policies:
    they are trained live for the page, so the config and the seeds
    stand in for a checkpoint identifier.
    """
    if fast:
        floors, horizons, mults = (90.0,), (30,), (4.0,)
        n_epochs, n_epochs_multi, n_paths = 2, 2, 64
    else:
        floors, horizons, mults = FLOORS, HORIZONS, MULTIPLIERS
        n_epochs, n_epochs_multi, n_paths = 40, 25, N_PATHS
    data_files = sorted(
        glob.glob(os.path.join(ROOT, "data", "spy_daily_*.csv")))
    if not fast:
        for ticker in ("tlt", "gld"):
            data_files.extend(sorted(glob.glob(
                os.path.join(ROOT, "data", f"{ticker}_daily_*.csv"))))
    data = {}
    for path in sorted(data_files):
        data[os.path.relpath(path, ROOT)] = _file_sha256(path)
    metrics = {}
    for rel in METRICS_FILES:
        path = os.path.join(ROOT, rel)
        metrics[rel] = _file_sha256(path) if os.path.exists(path) else "absent"
    return {
        "generator": "scripts/make_dashboard.py",
        "fast": bool(fast),
        "config": {
            "floors": list(floors),
            "horizons": list(horizons),
            "multipliers": list(mults),
            "n_epochs": n_epochs,
            "n_epochs_multi": n_epochs_multi,
            "n_paths": n_paths,
            "seeds": {"policy": 7, "simulation": 999, "multi": 11},
        },
        "data": data,
        "metrics": metrics,
        "commit": _git_commit(),
    }


def _dropdown(labels):
    buttons = []
    n = len(labels)
    for i, lab in enumerate(labels):
        visible = [False] * n
        visible[i] = True
        buttons.append(
            dict(label=lab, method="update",
                 args=[{"visible": visible},
                       {"title": lab}])
        )
    return [dict(buttons=buttons, direction="down", showactive=True,
                 x=0.0, y=1.12, xanchor="left", yanchor="top")]


def policy_figure(policies):
    """Heatmaps of risky weight over (cushion, time left), one per floor."""
    fig = go.Figure()
    labels = []
    floors = sorted(policies)
    for floor in floors:
        g = D.policy_weight_grid(policies[floor])
        fig.add_trace(
            go.Heatmap(
                z=g["weights"], x=g["taus"], y=g["cushions"],
                colorscale="Viridis", zmin=0, zmax=1,
                colorbar=dict(title="risky weight"),
                hovertemplate=("time left %{x:.0%}<br>"
                               "cushion %{y:.0%}<br>"
                               "weight %{z:.2f}<extra></extra>"),
                visible=(floor == 90.0),
            )
        )
        labels.append(f"Floor {floor:.0f}")
    fig.update_layout(
        title="Floor 90", updatemenus=_dropdown(labels),
        xaxis_title="fraction of time left",
        yaxis_title="cushion (fraction of wealth above the floor)",
        height=520,
    )
    return fig


def wealth_figure(sims):
    """Percentile bands for learned vs classic, selectable by setting."""
    fig = go.Figure()
    labels = []
    first = True
    combos = sorted(sims, key=lambda k: (k[0], k[1]))
    for horizon, mult in combos:
            sim = sims[(horizon, mult)]
            days = np.arange(horizon + 1)
            for name, color in (("learned", "#1f77b4"),
                                ("classic", "#ff7f0e")):
                s = sim[name]
                fig.add_trace(go.Scatter(
                    x=days, y=s["p50"], mode="lines",
                    name=f"{name} median", line=dict(color=color, width=2),
                    visible=first,
                    hovertemplate="day %{x}<br>wealth %{y:.1f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([days, days[::-1]]),
                    y=np.concatenate([s["p90"], s["p10"][::-1]]),
                    fill="toself", fillcolor=color, opacity=0.18,
                    line=dict(width=0), name=f"{name} p10-p90",
                    visible=first, hoverinfo="skip",
                ))
            labels.append(f"{horizon}d, classic m={mult:g}")
            first = False
    default = f"{63}d, classic m={4.0:g}"
    fig.update_layout(
        title=default if default in labels else labels[0],
        updatemenus=_dropdown(labels),
        xaxis_title="day", yaxis_title="wealth",
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def multi_figure(mgrid):
    """Multi-asset allocation: average weights and weights vs cushion."""
    labels = mgrid["labels"]
    avg_w = mgrid["weights"].mean(axis=(0, 1))
    mid = mgrid["weights"][:, len(mgrid["taus"]) // 2, :]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=avg_w, name="avg weight",
                         marker_color=["#1f77b4", "#ff7f0e", "#2ca02c",
                                       "#7f7f7f"]))
    fig.update_layout(title="Average allocation (SPY/TLT/GLD + cash)",
                      yaxis_title="weight", height=420)
    fig2 = go.Figure()
    for k, lab in enumerate(labels):
        fig2.add_trace(go.Scatter(
            x=mgrid["cushions"], y=mid[:, k], mode="lines",
            name=lab, stackgroup="one",
            hovertemplate="cushion %{x:.0%}<br>weight %{y:.2f}<extra></extra>",
        ))
    fig2.update_layout(
        title="Allocation vs cushion, halfway through the horizon",
        xaxis_title="cushion (fraction of wealth above the floor)",
        yaxis_title="weight", height=420,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig, fig2


def _metrics_table(title, rows):
    fig = go.Figure(data=[go.Table(
        header=dict(values=["strategy", "E[wealth]", "Std", "P(breach)"],
                    fill_color="#20303f", font_color="white", align="left"),
        cells=dict(values=[[r[0] for r in rows],
                            [f"{r[1]:.2f}" for r in rows],
                            [f"{r[2]:.2f}" for r in rows],
                            [f"{r[3]:.3f}" for r in rows]],
                   align="left"),
    )])
    fig.update_layout(title=title, height=280,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def results_figures(found):
    """One table figure per results section, from runs/ metrics."""
    figs = []
    if not found:
        fig = go.Figure()
        fig.add_annotation(
            text=("No metrics on disk yet. Run<br>"
                  "python -m dhl.experiments.cppi_learned<br>"
                  "python -m dhl.experiments.cppi_multi"),
            showarrow=False, font=dict(size=14))
        fig.update_layout(height=220, title="Repo results")
        return [fig]
    single = found.get("single_asset", {})
    for hz in ("30d", "63d"):
        if hz in single:
            rows = [(n, m["mean_wealth"], m["std_wealth"],
                     m["p_breach_terminal"])
                    for n, m in single[hz].items()]
            figs.append(_metrics_table(
                f"Single asset (SPY), {hz}, 20k held-out paths", rows))
    multi = found.get("multi_asset", {})
    for hz in ("30d", "63d"):
        if hz in multi.get("multi_asset", {}):
            rows = [(n, m["mean_wealth"], m["std_wealth"],
                     m["p_breach_terminal"])
                    for n, m in multi["multi_asset"][hz].items()]
            figs.append(_metrics_table(
                f"Multi asset (SPY/TLT/GLD + cash), {hz}, 10k paths", rows))
    dd = found.get("drawdown_63d", {})
    for mode, title in (("fixed", "Fixed floor"),
                        ("drawdown", "Drawdown floor")):
        if mode in dd:
            rows = [(n, m["mean_wealth"], m["std_wealth"],
                     m["p_breach_anytime"])
                    for n, m in dd[mode].items()]
            figs.append(_metrics_table(
                f"Drawdown experiment, 63d, 10k paths: {title}", rows))
    if "cross_decade" in multi:
        rows = [(n, m["mean_wealth"], m["std_wealth"],
                 m["p_breach_terminal"])
                for n, m in multi["cross_decade"].items()]
        figs.append(_metrics_table(
            "Cross-decade stress test: train pre-2020, test on 2020+", rows))
    return figs


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neural-cppi lab</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1000px; margin: 0 auto; padding: 24px; color: #1a1a1a; }}
h1 {{ margin-bottom: 4px; }} .sub {{ color: #555; margin-top: 0; }}
h2 {{ margin-top: 48px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
p {{ line-height: 1.55; }} .note {{ color: #555; font-size: 0.92em; }}
.plot {{ margin: 12px 0 32px 0; }}
footer {{ margin-top: 64px; color: #777; font-size: 0.85em;
          border-top: 1px solid #ddd; padding-top: 12px; }}
</style>
</head>
<body>
<h1>neural-cppi lab</h1>
<p class="sub">A neural network learns dynamic allocation across SPY, TLT,
GLD and cash, guarding a floor under wealth while costs eat every
rebalance. The demo policies below are small and trained live for this
page. The published numbers come from the full experiments.</p>

<h2>1. What the network learned</h2>
<p>The policy maps time left, cushion above the floor, and recent
volatility to a risky weight. Switch the floor and watch the mapping
shift: a higher floor means less cushion, so the network de-risks
earlier.</p>
<div class="plot">{policy}</div>

<h2>2. Learned vs classic CPPI</h2>
<p>Both strategies trade the same bootstrapped SPY paths, so the
difference is the allocation rule alone. Pick a horizon and a classic
multiplier. The classic rule never breaches on these paths. The learner
buys extra return with a small breach probability.</p>
<div class="plot">{wealth}</div>

<h2>3. Multi-asset allocation</h2>
<p>The 3-asset policy outputs softmax weights over SPY, TLT, GLD and
cash: long only, at most fully invested, by construction. The cash leg
swells as the cushion shrinks. That is the floor protection showing up
in the allocation itself.</p>
<div class="plot">{multi_bar}</div>
<div class="plot">{multi_area}</div>

<h2>4. Results from the full experiments</h2>
<p class="note">Read from <code>runs/</code> when this page was built.
Rerun the experiments to refresh.</p>
{results}

<footer>neural-cppi &middot; built from
<code>python scripts/make_dashboard.py</code> &middot; MIT</footer>
</body>
</html>
"""


def build(out_path, fast=False):
    if fast:
        floors, horizons, mults = (90.0,), (30,), (4.0,)
        n_epochs, n_epochs_multi, n_paths = 2, 2, 64
    else:
        floors, horizons, mults = FLOORS, HORIZONS, MULTIPLIERS
        n_epochs, n_epochs_multi, n_paths = 40, 25, N_PATHS

    print("training demo policies...", flush=True)
    policies = {f: D.train_demo_policy(floor=f, horizon_days=63,
                                       n_epochs=n_epochs, seed=7)
                for f in floors}
    pol_30 = D.train_demo_policy(floor=90.0, horizon_days=30,
                                 n_epochs=n_epochs, seed=7)

    print("simulating wealth paths...", flush=True)
    sims = {}
    for horizon in horizons:
        pol = policies[90.0] if horizon == 63 else pol_30
        for mult in mults:
            sims[(horizon, mult)] = D.simulate_wealth(
                pol, floor=90.0, horizon_days=horizon,
                multiplier=mult, n_paths=n_paths, seed=999)

    print("training multi-asset demo policy...", flush=True)
    mpol = (D.train_demo_multi(n_epochs=n_epochs_multi, seed=11)
            if not fast else None)
    mgrid = D.multi_weight_grid(mpol) if mpol is not None else None

    print("building figures...", flush=True)
    divs = {}
    divs["policy"] = plot(policy_figure(policies), output_type="div",
                         include_plotlyjs=True)
    divs["wealth"] = plot(wealth_figure(sims), output_type="div",
                         include_plotlyjs=False)
    if mgrid is not None:
        f1, f2 = multi_figure(mgrid)
        divs["multi_bar"] = plot(f1, output_type="div",
                                 include_plotlyjs=False)
        divs["multi_area"] = plot(f2, output_type="div",
                                  include_plotlyjs=False)
    else:
        divs["multi_bar"] = "<p class='note'>skipped in fast mode</p>"
        divs["multi_area"] = ""
    res_divs = "".join(
        plot(f, output_type="div", include_plotlyjs=False)
        for f in results_figures(D.repo_metrics())
    )
    html = PAGE.format(results=res_divs, **divs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_portfolio_html(html))
    lineage_path = os.path.splitext(out_path)[0] + "-lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as fh:
        json.dump(dashboard_lineage(fast), fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {lineage_path}", flush=True)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build docs/dashboard.html")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="docs/dashboard.html")
    args = ap.parse_args()
    build(args.out, fast=args.fast)



def _portfolio_html(raw):
    """Presentation-only transform of this run’s freshly calculated figures."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'neural-cppi', raw)[0]

if __name__ == "__main__":
    main()
