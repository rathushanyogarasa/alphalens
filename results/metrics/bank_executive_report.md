# AlphaLens Executive Diagnostic Report

Generated: 2026-03-11T15:47:46.357054+00:00
Commit: bcdd1c788040b1d8a35fa288da46c22caba52383
Data mode: offline_snapshot

## Classification
| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| vader | 0.5292 | 0.5283 | 0.4991 |
| finbert | 0.8431 | 0.8435 | 0.8338 |

## Baseline Backtest
| Model | Ann Return | Sharpe | Max DD | Win Rate |
|---|---:|---:|---:|---:|
| vader | -44.1% | -3.9036 | -14.4% | 45.8% |
| finbert | -26.2% | -1.7707 | -40.1% | 45.6% |

## Walk-Forward OOS
| Model | Scenario | Ann Return | Sharpe | IC | +OOS folds | Break-even bps |
|---|---|---:|---:|---:|---:|---:|
| vader | baseline | 5.8% | 0.2145 | 0.0975 | 100.0% | 0.00 |
| vader | improved_no_regime_uplift | -47.1% | -1.3726 | 0.0450 | 0.0% | 0.00 |
| finbert | baseline | -16.0% | -1.3580 | -0.0018 | 0.0% | 0.00 |
| finbert | improved_no_regime_uplift | -47.7% | -3.6524 | 0.0183 | 0.0% | 0.00 |

## Cost Robustness
| Model | Scenario | Sharpe | Ann Return |
|---|---|---:|---:|
| vader | cost_0.0bps | -0.4987 | -22.7% |
| vader | cost_15.0bps | -1.3726 | -47.1% |
| vader | cost_30.0bps | -2.2465 | -63.9% |
| vader | cost_7.5bps | -0.9356 | -36.1% |
| finbert | cost_0.0bps | 0.4350 | 11.5% |
| finbert | cost_15.0bps | -3.6524 | -47.7% |
| finbert | cost_30.0bps | -7.7401 | -75.5% |
| finbert | cost_7.5bps | -1.6087 | -23.6% |

## Predictive Edge Conclusion
- finbert: weak/unstable predictive edge (Sharpe=-3.652, IC=0.0183)
- vader: weak/unstable predictive edge (Sharpe=-1.373, IC=0.0450)
