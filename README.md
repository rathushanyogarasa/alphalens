# AlphaLens
### AI-Powered Stock Sentiment & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

AlphaLens is an end-to-end market intelligence pipeline that reads financial headlines, classifies sentiment with finance-tuned NLP, tests whether those signals historically translated into abnormal returns, and converts the output into practical BUY/HOLD/SELL recommendations for single names and full watchlists. It is designed to give investment teams a fast, auditable way to move from unstructured news flow to portfolio-relevant decisions.

## Architecture
```text
[Stage 1: Data Prep] --> [Stage 2: Data Sources] --> [Stage 3: Train]
           |                        |                      |
           v                        v                      v
      train/val/test         combined_news.csv        best_model/
           \_______________________|_____________________/ 
                                   v
                      [Stage 4: Evaluate Models]
                                   |
                                   v
                         [Stage 5: Back-testing]
                                   |
                                   v
                      [Stage 6: Keyword Analysis]
                                   |
                                   v
                  [Recommendations: Stock + Portfolio]
```

## Key Features
- Fine-tuned FinBERT sentiment classification
- Multi-source news ingestion
- Keyword event study with CAR analysis
- Real-time BUY/HOLD/SELL recommendations
- Portfolio watchlist scanner
- Professional CLI interface

## Installation
```bash
git clone https://github.com/yourusername/alphalens
cd alphalens
pip install -r requirements.txt
```

## Quick Start
```bash
python main.py --quick-test
python cli.py --ticker AAPL
python cli.py --portfolio
```

## Results
### Model performance comparison
| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| VADER | 0.000 | 0.000 | 0.000 |
| FinBERT | 0.000 | 0.000 | 0.000 |

### Back-test metrics
| Strategy | Sharpe | Annualised Return | Max Drawdown | Win Rate |
|---|---:|---:|---:|---:|
| FinBERT | 0.00 | 0.0% | 0.0% | 0.0% |
| VADER | 0.00 | 0.0% | 0.0% | 0.0% |

### Top keywords by CAR
| Keyword | Event Count | Avg CAR |
|---|---:|---:|
| earnings beat | 0 | +0.0% |
| raised guidance | 0 | +0.0% |
| sec investigation | 0 | -0.0% |

## Methodology
- Why FinBERT over VADER: FinBERT is trained on financial language and captures context like guidance, margin pressure, and filing language better than lexicon-only rules.
- How the event study works: For each keyword, the pipeline computes abnormal returns around event dates versus a market benchmark model, then aggregates to average CAR.
- How recommendation weights work: Final signal scores combine sentiment direction, source credibility, keyword historical impact (CAR), and recency decay.
- Limitations and EMH context: News can be priced quickly, so edge may decay; model signals should be used as decision support alongside valuation, risk, and execution constraints.

## Project Structure
```text
alphalens/
+-- data/
¦   +-- raw/
¦   +-- processed/
+-- src/
¦   +-- __init__.py
¦   +-- data_prep.py
¦   +-- data_sources.py
¦   +-- model.py
¦   +-- train.py
¦   +-- evaluate.py
¦   +-- backtest.py
¦   +-- keyword_analysis.py
¦   +-- stock_engine.py
¦   +-- portfolio_engine.py
+-- notebooks/
¦   +-- exploratory_analysis.ipynb
+-- results/
¦   +-- plots/
¦   +-- metrics/
¦   +-- best_model/
+-- config.py
+-- main.py
+-- cli.py
+-- setup.py
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## References
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
- Malo, P. et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance.

## License
MIT
