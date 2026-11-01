<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="Neural-CPPI: Learned allocation under portfolio floor constraints" width="1200"></p>

<h1 align="center">Neural-CPPI</h1>

<p align="center">
  <a href="https://github.com/FaiyazEnayet/neural-cppi/actions/workflows/ci.yml"><img src="https://github.com/FaiyazEnayet/neural-cppi/actions/workflows/ci.yml/badge.svg?label=CI&labelColor=24292f" alt="CI"></a>
  <a href="https://hub.xerj.org/contributors/FaiyazEnayet"><img src="https://hub.xerj.org/contributors/FaiyazEnayet/badge.svg" alt="XERJ Corpus Hub contributor"></a>
  <img src="https://img.shields.io/badge/license-MIT-c6a66b?labelColor=24292f" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?labelColor=24292f" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/built%20with-PyTorch%20%2B%20NumPy-301934?labelColor=24292f" alt="Built with PyTorch + NumPy">
</p>

<p align="center">
  <a href="https://faiyazenayet.github.io/neural-cppi/docs/dashboard.html"><img src="https://img.shields.io/badge/Open-Dashboard-f4edda?style=for-the-badge&labelColor=111215" alt="Open Dashboard"></a>
  <a href="#method"><img src="https://img.shields.io/badge/Explore-Method-414348?style=for-the-badge&labelColor=111215" alt="Explore Method"></a>
  <a href="#historical-tail-risk-challenge"><img src="https://img.shields.io/badge/Review-Historical%20Stress-c6a66b?style=for-the-badge&labelColor=111215" alt="Review Historical Stress"></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Run-Locally-1a1b1e?style=for-the-badge&labelColor=111215" alt="Run Locally"></a>
</p>

A research implementation of neural portfolio insurance. It compares learned allocation policies with classical CPPI and makes the trade between terminal wealth and floor protection inspectable.


<p align="center"><a href="https://faiyazenayet.github.io/neural-cppi/docs/dashboard.html"><img src="docs/demo.gif?v=showcase-20261017b" alt="Neural-CPPI demo"></a></p>


The dashboard separates return-seeking behavior from protection outcomes. Read the wealth comparison together with breach frequency, horizon and allocation policy: a stronger average result can still carry undesirable tail risk. The regime summaries provide a second check on whether conclusions depend on a particular return environment.

## Explore

- Compare wealth bands with the protection floor across source horizons and classical multipliers.
- Inspect learned allocation at different floor levels, then read the corresponding experimental tables.
- Review transaction-cost assumptions, drawdown constraints and cross-regime evidence alongside returns.

## Method

Historical daily returns generate block-bootstrap scenarios. A neural policy learns risky allocation or a long-only multi-asset mix. Evaluation separates synthetic experiments, resampled market paths and held-out regime tests.

## Evidence at a glance

| Scope | Recorded result or contract |
| :--- | :--- |
| 63-day synthetic learned-policy experiment | Mean terminal wealth 101.70 from initial wealth 100 |
| Same experiment | Terminal wealth standard deviation 3.63; observed terminal-floor breaches 0.3% |
| Interpretation | Finite-sample outcomes, not a capital-protection guarantee |

## Updating this project

Dashboard source language: **Python**. Generator: `scripts/make_dashboard.py`. The local interactive output is `docs/dashboard.html`.

## Run locally


```sh
python -m pip install -e ".[dev]"
python -m pip install -r portfolio_ui/requirements.txt
python -m pytest tests -q
python -m dhl.experiments.cppi_learned
python -m dhl.experiments.cppi_multi
python scripts/make_dashboard.py
```

## Technical design

The single-asset policy uses time remaining, cushion and volatility to choose risky exposure. The multi-asset version uses long-only portfolio weights and a drawdown-sensitive floor. A safety layer can cap total risky exposure while the learned policy chooses the asset mix. These are distinct experimental variants, so their results should not be combined into one headline return.

```mermaid
flowchart LR
  N0["Historical daily bars"] --> N1["Block bootstrap"]
  N1["Block bootstrap"] --> N2["Learned allocation"]
  N1["Block bootstrap"] --> N3["Classical CPPI"]
  N2["Learned allocation"] --> N4["Risk and return evaluation"]
  N3["Classical CPPI"] --> N4["Risk and return evaluation"]
```

## Evaluation protocol

The committed dashboard contains separate synthetic experiments, a historical-return bootstrap explorer and historical regime summaries. The bootstrap bands describe resampled scenarios, not a realized investment track record. Floor-breach rates must be read with the horizon, path count, transaction-cost assumptions and exact floor definition.

![Terminal wealth in the 63-day synthetic experiment](docs/results.png)

The illustrated 63-day experiment improves mean wealth over the classical policies while accepting some terminal-floor breaches. That is an observed tradeoff. It does not establish that the learned policy dominates CPPI in every market.

## Historical tail-risk challenge

This pre-registered stress evaluation tested the frozen shipped system on historical crash windows that played no part in training or model selection. The system under test was the learned allocation policy combined with its deterministic safety layer. Its comparator was classic CPPI at the same multiplier, **4.0**, on identical data and transaction costs of **20 bps per unit of turnover**. Initial wealth was **100**, with a floor of **90%** of initial wealth. A breach meant wealth below the floor at any daily close within a window.

The evaluation covered **38 windows**. The main set comprised **31 US windows, 1929–2022**, spanning prolonged collapses, sudden crashes, grinding bear markets, V-shaped shocks and a rate-shock bear market, at horizons of **252 or 504 trading days**. A second set comprised **7 windows** across Japan, Hong Kong, Germany and China. Windows before **1957** used a total-market research series as the index proxy. All windows used daily closes.

The aggregate spans two distinct uses within the improvement program: **25 sealed windows**, comprising **11 test, 7 held-out and 7 international windows**, were evaluated once after the freeze; **13 tuning-tier windows**, comprising **9 validation and 4 layer-tuning windows**, were used only to develop and judge improvement candidates. The shipped system was never trained or selected on any of these windows. On the sealed 25, it recorded **zero breaches**, worst minimum wealth **90.33**, and terminal wealth wins over classic CPPI in **15 of 18 sealed main-set windows** and **6 of 7 international windows**. The 38-window aggregate below is broader than this sealed evaluation.

The shipped system recorded **zero anytime breaches in 38 of 38 windows**, with worst observed minimum wealth **90.33**. **Classic CPPI also recorded zero breaches.** This protection result does not distinguish the shipped system from classic CPPI. On terminal wealth, the shipped system finished ahead in **28 of the 31 main-set windows**, but the typical paired advantage was modest: median **+0.19** wealth units and mean **+0.30**. Half of the wins were by **+0.25 or less**.

![Anytime floor breaches by configuration. Both complete systems recorded 0 of 38; the learned policy without its safety layer breached in 8 of 11 Great Depression subset windows.](docs/tail-risk-breaches.png)

In the separate **11-window Great Depression ablation subset**, removing the safety layer left the learned policy with breaches in **8 of 11 windows**. The protection record therefore belongs to the combined policy and safety layer, which is the configuration the repository ships.

The principal limitation was post-trough recovery. In the window starting at the **1932 trough**, classic CPPI reached terminal wealth **153.38**, against **126.14** for the shipped system. Across the recovery windows, classic CPPI held a mean risky weight of **0.500**, against **0.151** for the learned policy. The learned policy does not commit its full permitted budget after a crash.

![Distribution of all 31 main-set paired terminal wealth differences, shipped minus classic. The median is +0.19; the 1932 trough-start difference of -27.24 remains visible.](docs/tail-risk-terminal.png)

The windows overlap by construction: paired horizons share start dates. The **38 windows are not 38 independent trials**, and the observed zero-breach record is not a universal floor-protection guarantee.

Four candidate improvements, multi-horizon training, an anytime shortfall penalty, a rebound feature and a recovery spending rule, were tuned only within the separate validation and layer-tuning tiers, frozen, and judged once against sealed windows under a pre-registered adoption rule. None met the rule; the shipped system is unchanged. One early evaluation of the penalty candidate used a mis-signed objective term. It was rerun with the sign corrected under identical budgets, data and seeds, and the rejection stood.

The complete study is retained as a private report. Its record is fixed by SHA-256 `760c45fde1f557b4b257fb39f85761e1af6f81cfa0c04d92a99df3dcc6ca303a`. This section publishes summary results and aggregate charts only.


## Code and data

`src/dhl/` contains the policies, market samplers, risk calculations and experiments. `data/` contains the committed daily market observations. `scripts/make_dashboard.py` builds the HTML output.


## Limitations

- Resampling cannot create a market regime absent from the source history.
- Daily CPPI is exposed to gap risk, discrete rebalancing and transaction costs. Zero observed breaches do not prove zero risk.
- Ablation conclusions need repeated training seeds and uncertainty intervals; a single crisis is a limited stress test.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
