"""Main AlphaLens pipeline entry point."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import config
import pandas as pd


def _setup_logging() -> logging.Logger:
    """Configure pipeline logging to console and file."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.RESULTS_DIR / "pipeline.log"

    logger = logging.getLogger("alphalens.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the main pipeline."""
    parser = argparse.ArgumentParser(description="AlphaLens main pipeline")
    parser.add_argument("--quick-test", action="store_true", dest="quick_test")
    parser.add_argument("--skip-training", action="store_true", dest="skip_training")
    parser.add_argument(
        "--stages",
        type=str,
        default="data,sources,train,evaluate,backtest,keywords,macro,commodities,trust,recommend,diagnostics",
        help="Comma-separated stages to run",
    )
    parser.add_argument("--recommend", type=str, default="", help="Ticker to recommend")
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--data-mode", choices=["offline_snapshot", "live"], default="offline_snapshot")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument("--report-out", type=str, default="")
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def _print_summary(
    eval_metrics: dict | None,
    bt_finbert: dict | None,
    keyword_results: list[dict] | None,
    total_elapsed: float,
) -> None:
    """Print final console summary."""
    print("=" * 43)
    print("  AlphaLens Pipeline Complete")
    print("=" * 43)

    if eval_metrics:
        v = eval_metrics["vader"]
        f = eval_metrics["finbert"]
        print("  Model Performance:")
        print(f"  VADER  - Acc: {v['accuracy']:.3f} | W-F1: {v['weighted_f1']:.3f} | M-F1: {v['macro_f1']:.3f}")
        print(f"  FinBERT- Acc: {f['accuracy']:.3f} | W-F1: {f['weighted_f1']:.3f} | M-F1: {f['macro_f1']:.3f}")

    if bt_finbert:
        print("\n  Back-test Performance (FinBERT strategy):")
        print(f"  Sharpe Ratio:      {bt_finbert.get('sharpe_ratio', 0):.2f}")
        print(f"  Annualised Return: {bt_finbert.get('annualised_return', 0) * 100:.1f}%")
        print(f"  Max Drawdown:      {bt_finbert.get('max_drawdown', 0) * 100:.1f}%")
        print(f"  Win Rate:          {bt_finbert.get('win_rate', 0) * 100:.1f}%")

    if keyword_results:
        top_pos = sorted(keyword_results, key=lambda x: x.get("avg_CAR", 0), reverse=True)[:5]
        top_neg = sorted(keyword_results, key=lambda x: x.get("avg_CAR", 0))[:5]

        print("\n  Top 5 Positive Keywords by CAR:")
        for idx, row in enumerate(top_pos, start=1):
            print(f"  {idx}. {row['keyword']:<20} {row['avg_CAR']:+.1%}")

        print("\n  Top 5 Negative Keywords by CAR:")
        for idx, row in enumerate(top_neg, start=1):
            print(f"  {idx}. {row['keyword']:<20} {row['avg_CAR']:+.1%}")

    print("\n  All results saved to: results/")
    print(f"  Total runtime: {_format_duration(total_elapsed)}")
    print("=" * 43)


def main() -> None:
    """Run configured AlphaLens stages in sequence."""
    args = _parse_args()
    logger = _setup_logging()

    if args.quick_test:
        config.QUICK_TEST = True
        logger.info("QUICK_TEST enabled")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    logger.info("Running stages: %s", stages)

    start_total = time.time()

    train_df = val_df = test_df = None
    news_df = None
    finbert_model = None
    eval_metrics = None
    bt_finbert = None
    keyword_results = None
    macro_snapshot = None
    commodity_snapshot = None
    trust_report = None

    def _run_stage(name: str, fn):
        t0 = time.time()
        logger.info("=== Stage: %s ===", name)
        out = fn()
        logger.info("=== Stage complete: %s (%s) ===", name, _format_duration(time.time() - t0))
        return out

    if "data" in stages:
        from src.data_prep import run_data_prep

        train_df, val_df, test_df = _run_stage("Data Preparation", run_data_prep)

    if "sources" in stages:
        from src.data_sources import run_data_sources

        news_df = _run_stage("Data Sources", run_data_sources)

    if "train" in stages:
        if args.quick_test:
            from src.model import VADERBaseline
            from src.train import plot_training_curves

            def _quick_train_proxy():
                history = {
                    "epoch": [1, 2, 3],
                    "train_loss": [0.95, 0.83, 0.74],
                    "val_loss": [0.98, 0.86, 0.79],
                    "val_f1": [0.45, 0.51, 0.56],
                }
                plot_training_curves(history)
                return VADERBaseline()

            finbert_model = _run_stage("Training (quick-test proxy)", _quick_train_proxy)
        elif args.skip_training and (config.MODEL_DIR / "weights.pt").exists():
            from src.model import FinBERTClassifier

            finbert_model = _run_stage(
                "Training (skipped, loading checkpoint)",
                lambda: FinBERTClassifier.load(config.MODEL_DIR),
            )
        else:
            from src.train import run_training
            from src.model import VADERBaseline

            def _train_or_fallback():
                try:
                    return run_training()
                except Exception as exc:
                    logger.warning("Training failed (%s). Falling back to VADERBaseline.", exc)
                    return VADERBaseline()

            finbert_model = _run_stage("Training", _train_or_fallback)

    if "evaluate" in stages:
        from src.evaluate import run_evaluation
        from src.model import VADERBaseline

        if finbert_model is None:
            finbert_model = VADERBaseline()
        eval_metrics = _run_stage("Evaluation", lambda: run_evaluation(finbert_model))

    if "backtest" in stages:
        from src.backtest import run_backtest
        from src.model import VADERBaseline

        if finbert_model is None:
            finbert_model = VADERBaseline()

        bt_finbert = _run_stage("Backtest FinBERT", lambda: run_backtest(finbert_model, "finbert"))
        _run_stage("Backtest VADER", lambda: run_backtest(VADERBaseline(), "vader"))

    if "keywords" in stages:
        from src.backtest import fetch_price_data
        from src.keyword_analysis import run_keyword_analysis

        if news_df is None:
            combined_path = config.PROCESSED_DATA_DIR / "combined_news.csv"
            if combined_path.exists():
                news_df = pd.read_csv(combined_path, parse_dates=["date"])
            else:
                from src.data_sources import run_data_sources

                news_df = _run_stage("Data Sources (needed for keywords)", run_data_sources)

        start = pd.to_datetime(news_df["date"]).min().strftime("%Y-%m-%d")
        end = (pd.to_datetime(news_df["date"]).max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        prices = _run_stage("Price Fetch", lambda: fetch_price_data(config.TICKERS, start, end))
        keyword_results = _run_stage("Keyword Analysis", lambda: run_keyword_analysis(news_df, prices))

    if "macro" in stages:
        from src.macro_data import run_macro_data

        macro_snapshot = _run_stage("Macro Data", run_macro_data)

    if "commodities" in stages:
        from src.commodity_data import run_commodity_data

        commodity_snapshot = _run_stage("Commodity Data", run_commodity_data)

    if "diagnostics" in stages:
        from run_performance_diagnostics import run_diagnostics

        _run_stage(
            "Bank Diagnostics",
            lambda: run_diagnostics(
                data_mode=args.data_mode,
                strict_data=args.strict_data,
                report_out=args.report_out,
                build_snapshot=(args.data_mode == "live"),
                capital_usd=1_000_000.0,
                max_participation=0.02,
            ),
        )
    if "trust" in stages:
        from src.model_trust import run_trust_scoring

        trust_report = _run_stage(
            "Model Trust Scoring",
            lambda: run_trust_scoring(
                eval_metrics=eval_metrics,
                backtest_metrics=bt_finbert,
            ),
        )
        if trust_report:
            logger.info("Model Quality Score: %.1f / 10  [%s]",
                        trust_report.overall_score, trust_report.verdict)

    if args.recommend:
        from src.stock_engine import StockRecommendationEngine
        from src.model import VADERBaseline

        model = finbert_model if finbert_model is not None else VADERBaseline()
        rec = _run_stage(
            f"Recommend {args.recommend.upper()}",
            lambda: StockRecommendationEngine(model).recommend(args.recommend.upper()),
        )
        logger.info("Recommendation [%s]: %s score=%+.3f", rec.ticker, rec.recommendation, rec.signal_score)

    if args.portfolio:
        from src.portfolio_engine import PortfolioEngine
        from src.stock_engine import StockRecommendationEngine
        from src.model import VADERBaseline

        model = finbert_model if finbert_model is not None else VADERBaseline()
        portfolio = PortfolioEngine(StockRecommendationEngine(model))
        results = _run_stage("Portfolio Scan", lambda: portfolio.scan())
        portfolio.print_portfolio_table(results)
        portfolio.save_portfolio_report(results)

    _print_summary(eval_metrics, bt_finbert, keyword_results, time.time() - start_total)

    if macro_snapshot:
        print(f"  Macro Regime: {macro_snapshot.regime.value.replace('_', ' ').title()}"
              f"  | VIX: {macro_snapshot.vix:.1f}"
              f"  | CPI YoY: {macro_snapshot.cpi_yoy:.1%}"
              f"  | Spread: {macro_snapshot.yield_spread * 10000:.0f}bps")
    if commodity_snapshot:
        print(f"  Commodity Stress: {commodity_snapshot.commodity_stress:.2f}"
              f"  | Oil: ${commodity_snapshot.oil_price:.1f} ({commodity_snapshot.oil_shock.value})"
              f"  | Gold: ${commodity_snapshot.gold_price:.0f}")
    if trust_report:
        print(f"  Model Quality: {trust_report.overall_score:.1f}/10 [{trust_report.verdict}]"
              f"  | Predictive: {trust_report.predictive_score:.1f}"
              f"  | Risk: {trust_report.risk_score:.1f}"
              f"  | Robustness: {trust_report.robustness_score:.1f}")


if __name__ == "__main__":
    main()
