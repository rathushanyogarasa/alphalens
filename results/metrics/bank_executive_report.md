# AlphaLens Executive Diagnostic Report

Generated: 2026-03-16T00:41:09.322072+00:00
Commit: dc7e91d285a1787edf7c02ed925f9af34f0a3f3c
Data mode: offline_snapshot

## Classification
| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| vader | 1.0000 | 1.0000 | 1.0000 |
| finbert | 0.6667 | 0.5556 | 0.5556 |

## Baseline Backtest
| Model | Ann Return | Sharpe | Max DD | Win Rate |
|---|---:|---:|---:|---:|
| vader | -44.1% | -3.9036 | -14.4% | 45.8% |
| finbert | -26.2% | -1.7707 | -40.1% | 45.6% |

## Walk-Forward OOS
| Model | Scenario | Ann Return | Sharpe | IC | +OOS folds | Break-even bps |
|---|---|---:|---:|---:|---:|---:|
| vader | baseline | -4.5% | -1.4826 | 0.0936 | 0.0% | 0.00 |
| vader | improved_no_regime_uplift | -47.1% | -1.3726 | 0.0450 | 0.0% | 0.00 |
| finbert | baseline | -29.7% | -2.7960 | -0.0018 | 0.0% | 0.00 |
| finbert | improved | -44.4% | -3.9041 | 0.0360 | 0.0% | 0.00 |

## Cost Robustness
| Model | Scenario | Sharpe | Ann Return |
|---|---|---:|---:|
| vader | cost_0.0bps | -0.4987 | -22.7% |
| vader | cost_15.0bps | -1.3726 | -47.1% |
| vader | cost_30.0bps | -2.2465 | -63.9% |
| vader | cost_7.5bps | -0.9356 | -36.1% |
| finbert | cost_0.0bps | 0.8674 | 18.5% |
| finbert | cost_15.0bps | -3.9041 | -44.4% |
| finbert | cost_30.0bps | -8.6754 | -74.0% |
| finbert | cost_7.5bps | -1.5180 | -18.8% |

## Predictive Edge Conclusion
- finbert: weak/unstable predictive edge (Sharpe=-3.904, IC=0.0360)
- vader: weak/unstable predictive edge (Sharpe=-1.373, IC=0.0450)
