from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.backtest import calculate_portfolio_metrics, calculate_strategy_returns, fetch_price_data
from src.data_prep import load_splits
from src.evaluate import evaluate_model
from src.factor_engine import build_factor_signals, combine_factors, fetch_volume_data
from src.longshort_engine import _aggregate_signals, run_longshort_backtest, run_longshort_from_signals
from src.model import FinBERTClassifier, SIGNAL_MAP, VADERBaseline
from src.sentiment_cache import CachedPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('bank_diagnostics')


def _set_seed(seed=42):
    np.random.seed(seed)


def _snapshot_paths() -> dict[str, Path]:
    root = config.RESEARCH_DATA_DIR
    return {
        'news': root / 'news.csv',
        'prices': root / 'prices.csv',
        'volumes': root / 'volumes.csv',
        'benchmark': root / 'benchmark.csv',
        'risk_free': root / 'risk_free.csv',
        'meta': root / 'metadata.json',
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(config.BASE_DIR), text=True).strip()
    except Exception:
        return 'unknown'


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.exists() and p.is_file():
            h.update(str(p).encode('utf-8'))
            h.update(_hash_file(p).encode('utf-8'))
    return h.hexdigest()


def _load_news_live() -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / 'combined_news.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df


def _save_snapshot(news: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, benchmark: pd.DataFrame, risk_free: pd.DataFrame):
    p = _snapshot_paths()
    config.RESEARCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    news.to_csv(p['news'], index=False)
    prices.to_csv(p['prices'])
    volumes.to_csv(p['volumes'])
    benchmark.to_csv(p['benchmark'])
    risk_free.to_csv(p['risk_free'], index=False)
    p['meta'].write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'coverage_start': str(news['date'].min().date()) if not news.empty else 'n/a',
        'coverage_end': str(news['date'].max().date()) if not news.empty else 'n/a',
        'n_headlines': int(len(news)),
    }, indent=2), encoding='utf-8')


def _load_snapshot(strict: bool):
    p = _snapshot_paths()
    missing = [k for k, v in p.items() if k != 'meta' and not v.exists()]
    if missing:
        raise RuntimeError(f'missing snapshot files: {", ".join(missing)}')
    news = pd.read_csv(p['news'], parse_dates=['date'])
    prices = pd.read_csv(p['prices'], index_col=0, parse_dates=True)
    volumes = pd.read_csv(p['volumes'], index_col=0, parse_dates=True)
    benchmark = pd.read_csv(p['benchmark'], index_col=0, parse_dates=True)
    risk_free = pd.read_csv(p['risk_free'], parse_dates=['date'])
    news['date'] = pd.to_datetime(news['date']).dt.normalize()
    if strict and (news.empty or prices.empty):
        raise RuntimeError('snapshot is empty in strict mode')
    return news, prices, volumes, benchmark, risk_free


def _load_data(data_mode: str, strict_data: bool, build_snapshot: bool):
    if data_mode == 'offline_snapshot':
        return _load_snapshot(strict_data)

    news = _load_news_live()
    start = news['date'].min().strftime('%Y-%m-%d')
    end = (news['date'].max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    prices = fetch_price_data(config.TICKERS, start=start, end=end, allow_synthetic=not strict_data)
    if prices.empty:
        raise RuntimeError('live price fetch failed')

    volumes = fetch_volume_data(config.TICKERS, start=start, end=end)
    benchmark = fetch_price_data([config.MARKET_BENCHMARK], start=start, end=end, allow_synthetic=not strict_data)
    risk_free = pd.DataFrame({'date': prices.index, 'risk_free_daily': config.RISK_FREE_RATE / 252})

    if build_snapshot:
        _save_snapshot(news, prices, volumes, benchmark, risk_free)

    return news, prices, volumes, benchmark, risk_free


def _load_finbert_or_fallback():
    checkpoint = config.MODEL_DIR / 'weights.pt'
    if checkpoint.exists():
        try:
            import torch
            model = FinBERTClassifier(model_name=str(config.MODEL_DIR))
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state)
            model.eval()
            return model, 'finbert'
        except Exception as exc:
            logger.warning('FinBERT load failed (%s); fallback to VADER', exc)
    return VADERBaseline(), 'vader_fallback'


def _score_news(news: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
    predictor = CachedPredictor(model, model_name)
    out = predictor.predict_dataframe(news)
    out['date'] = pd.to_datetime(out['date']).dt.normalize()
    out['signal_normal'] = out['label_name'].map(SIGNAL_MAP).fillna(0)
    return out


def _coverage_ratio(prices: pd.DataFrame) -> float:
    if prices.empty:
        return 0.0
    return float(prices.notna().sum().sum()) / float(prices.shape[0] * prices.shape[1])


def _event_backtest(scored: pd.DataFrame, prices: pd.DataFrame, model_name: str):
    conf = scored[scored['confidence'] >= config.CONFIDENCE_THRESHOLD].copy()
    if conf.empty:
        conf = scored.copy()
    signals = conf.groupby(['date', 'ticker']).agg(signal=('signal_normal', 'mean')).reset_index()
    returns = calculate_strategy_returns(signals, prices)
    return calculate_portfolio_metrics(returns, model_name=model_name)


def _apply_liquidity_filter(sig_df: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame):
    if sig_df.empty or prices.empty or volumes.empty:
        return sig_df
    dv = (prices * volumes).stack().rename('dollar_vol').reset_index()
    dv.columns = ['date', 'ticker', 'dollar_vol']
    out = sig_df.merge(dv, on=['date', 'ticker'], how='left')
    q20 = out.groupby('date')['dollar_vol'].transform(lambda s: s.quantile(0.2) if s.notna().any() else np.nan)
    out = out[(out['dollar_vol'].isna()) | (out['dollar_vol'] >= q20)].drop(columns=['dollar_vol'])
    return out


def _apply_tradeability(trades: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, capital_usd: float, max_participation: float):
    if trades.empty or prices.empty or volumes.empty:
        out = trades.copy()
        out['participation_proxy'] = np.nan
        return out
    dv = (prices * volumes).stack().rename('dollar_vol').reset_index()
    dv.columns = ['entry_date', 'ticker', 'dollar_vol']
    out = trades.merge(dv, on=['entry_date', 'ticker'], how='left')
    out['participation_proxy'] = out['weight'].abs() * capital_usd / out['dollar_vol'].replace(0, np.nan)
    scale = np.minimum(1.0, max_participation / out['participation_proxy'].replace(0, np.nan)).fillna(1.0)
    out['ls_return'] = out['ls_return'] * scale
    return out


def _metrics_from_trades(trades: pd.DataFrame):
    if trades.empty:
        return {}
    net = trades.groupby('rebalance_date')['ls_return'].sum().sort_index()
    pp = 252.0 if len(net) < 2 else 252.0 / max(1.0, float(np.median([max(1, len(pd.bdate_range(net.index[i - 1], net.index[i])) - 1) for i in range(1, len(net))])))
    total = float((1 + net).prod() - 1)
    ann = float((1 + total) ** (pp / max(1, len(net))) - 1)
    rf = config.RISK_FREE_RATE / pp
    excess = net - rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(pp)) if excess.std() > 0 else 0.0
    eq = (1 + net).cumprod()
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) else 0.0
    win = float((net > 0).mean()) if len(net) else 0.0
    ic = float(trades['signal'].corr(trades['period_return'], method='spearman')) if len(trades) > 10 else np.nan
    return {
        'annualised_return': round(ann, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(dd, 4),
        'win_rate': round(win, 4),
        'ic': round(ic, 4) if pd.notna(ic) else np.nan,
        'avg_turnover': np.nan,
        'trade_count': int(len(trades)),
        'participation_proxy_mean': round(float(trades['participation_proxy'].mean()), 6) if 'participation_proxy' in trades else np.nan,
    }


def _run_strategy(scored: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, conf: float, q: float, hold: int, variant: str, regime: bool, tc: float, slip: float, capital: float, max_participation: float):
    if variant == 'sentiment':
        trades, m = run_longshort_backtest(scored, prices, hold_days=hold, quantile_cutoff=q, confidence_threshold=conf, signal_col='signal_normal', weighting='equal', macro_vix_filter=regime, tc_bps=tc, slip_bps=slip)
    else:
        raw = _aggregate_signals(scored, 'signal_normal', conf, confidence_weighted=True, max_headlines_per_day=3)
        raw = _apply_liquidity_filter(raw, prices, volumes)
        fac = build_factor_signals(raw, prices, volumes if not volumes.empty else None)
        fac = combine_factors(fac, weights={'sentiment': 0.6, 'momentum': 0.3, 'volatility': 0.1, 'liquidity': 0.0})
        trades, m = run_longshort_from_signals(fac, prices, signal_col='combined_score', hold_days=hold, quantile_cutoff=q, tc_bps=tc, slip_bps=slip, weighting='equal', macro_vix_filter=regime)
    if trades.empty:
        return trades, {}
    trades = _apply_tradeability(trades, prices, volumes, capital, max_participation)
    m2 = _metrics_from_trades(trades)
    m2.update({'hold_days': hold, 'quantile_cutoff': q, 'confidence_threshold': conf, 'variant': variant, 'regime_filter': regime})
    return trades, m2


def _calibrate(train_scored: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, capital: float, max_participation: float):
    best = None
    for conf, q, hold, variant, regime in itertools.product([0.60, 0.75], [0.15, 0.25], [1, 2], ['sentiment', 'ensemble'], [False, True]):
        _, m = _run_strategy(train_scored, prices, volumes, conf, q, hold, variant, regime, config.TRANSACTION_COST_BPS, config.SLIPPAGE_BPS, capital, max_participation)
        if not m:
            continue
        sh = float(m.get('sharpe_ratio', np.nan))
        if not np.isfinite(sh):
            continue
        score = sh - 2.0 * max(0.0, float(m.get('participation_proxy_mean', 0.0) or 0.0) - max_participation)
        cand = {'confidence_threshold': conf, 'quantile_cutoff': q, 'hold_days': hold, 'variant': variant, 'regime_filter': regime, 'score': score, 'train_sharpe': sh}
        if best is None or cand['score'] > best['score']:
            best = cand
    return best


def _aggregate_oos(trades: pd.DataFrame, period_df: pd.DataFrame, model: str):
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for scenario in sorted(trades['scenario'].unique()):
        t = trades[trades['scenario'] == scenario]
        m = _metrics_from_trades(t)
        p = period_df[period_df['scenario'] == scenario]
        rows.append({
            'model': model,
            'scenario': scenario,
            'annualised_return': m.get('annualised_return', np.nan),
            'sharpe_ratio': m.get('sharpe_ratio', np.nan),
            'max_drawdown': m.get('max_drawdown', np.nan),
            'win_rate': m.get('win_rate', np.nan),
            'trade_count': int(len(t)),
            'avg_turnover': float(p['avg_turnover'].mean()) if ('avg_turnover' in p and not p.empty) else np.nan,
            'ic': float(p['ic'].mean()) if ('ic' in p and not p.empty) else np.nan,
            'ic_stability': float(p['ic'].std()) if ('ic' in p and len(p) > 1) else np.nan,
            'oos_fold_count': int(p['period'].nunique()) if ('period' in p and not p.empty) else 0,
            'pct_positive_oos_folds': float((p['sharpe_ratio'] > 0).mean()) if ('sharpe_ratio' in p and not p.empty) else np.nan,
            'break_even_cost_bps': np.nan,
            'data_coverage_ratio': np.nan,
        })
    return pd.DataFrame(rows)


def _walk_forward(scored: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, model: str, capital: float, max_participation: float):
    w = scored.copy(); w['period'] = w['date'].dt.to_period('Q')
    periods = sorted(w['period'].unique())
    if len(periods) <= 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trades, prows, choices, ablation = [], [], [], []
    start_idx = max(4, len(periods) - 4)
    for i in range(start_idx, len(periods)):
        test_p = periods[i]
        tr = w[w['period'].isin(periods[:i])].drop(columns=['period'])
        te = w[w['period'] == test_p].drop(columns=['period'])
        if tr.empty or te.empty:
            continue
        best = _calibrate(tr, prices, volumes, capital, max_participation)
        if not best:
            continue
        choices.append({'model': model, 'test_period': str(test_p), **best})

        bt, bm = _run_strategy(te, prices, volumes, config.CONFIDENCE_THRESHOLD, 0.20, 2, 'sentiment', False, config.TRANSACTION_COST_BPS, config.SLIPPAGE_BPS, capital, max_participation)
        if not bt.empty and bm:
            x = bt.copy(); x['scenario'] = 'baseline'; x['period'] = str(test_p); trades.append(x)
            prows.append({'model': model, 'period': str(test_p), 'scenario': 'baseline', **bm})

        it, im = _run_strategy(te, prices, volumes, best['confidence_threshold'], best['quantile_cutoff'], best['hold_days'], best['variant'], best['regime_filter'], config.TRANSACTION_COST_BPS, config.SLIPPAGE_BPS, capital, max_participation)
        if not it.empty and im:
            x = it.copy(); x['scenario'] = 'improved'; x['period'] = str(test_p); trades.append(x)
            prows.append({'model': model, 'period': str(test_p), 'scenario': 'improved', **im})

        for label, variant, regime in [('ablation_sentiment', 'sentiment', False), ('ablation_ensemble', 'ensemble', False), ('ablation_ensemble_regime', 'ensemble', True)]:
            _, am = _run_strategy(te, prices, volumes, best['confidence_threshold'], best['quantile_cutoff'], best['hold_days'], variant, regime, config.TRANSACTION_COST_BPS, config.SLIPPAGE_BPS, capital, max_participation)
            if am:
                ablation.append({'model': model, 'period': str(test_p), 'scenario': label, **am})

    trades_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    period_df = pd.DataFrame(prows)
    choices_df = pd.DataFrame(choices)
    ablation_df = pd.DataFrame(ablation)
    summary = _aggregate_oos(trades_df, period_df, model)

    if not ablation_df.empty and not summary.empty:
        g = ablation_df.groupby('scenario')['sharpe_ratio'].mean()
        if 'ablation_ensemble' in g and 'ablation_ensemble_regime' in g and g['ablation_ensemble_regime'] <= g['ablation_ensemble']:
            summary.loc[summary['scenario'] == 'improved', 'scenario'] = 'improved_no_regime_uplift'

    return summary, period_df, choices_df, ablation_df


def _cost_robustness(scored: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, model: str, choices_df: pd.DataFrame, capital: float, max_participation: float):
    if choices_df.empty:
        return pd.DataFrame()
    w = scored.copy(); w['period'] = w['date'].dt.to_period('Q')
    t_all, p_all = [], []

    for _, r in choices_df.iterrows():
        te = w[w['period'].astype(str) == str(r['test_period'])].drop(columns=['period'])
        if te.empty:
            continue
        for bps in [0.0, 7.5, 15.0, 30.0]:
            tc, sl = bps * 2.0 / 3.0, bps / 3.0
            td, m = _run_strategy(te, prices, volumes, float(r['confidence_threshold']), float(r['quantile_cutoff']), int(r['hold_days']), str(r['variant']), bool(r['regime_filter']), tc, sl, capital, max_participation)
            if td.empty or not m:
                continue
            name = f'cost_{bps:.1f}bps'
            x = td.copy(); x['scenario'] = name; x['period'] = str(r['test_period']); t_all.append(x)
            p_all.append({'model': model, 'period': str(r['test_period']), 'scenario': name, **m})

    if not t_all:
        return pd.DataFrame()
    out = _aggregate_oos(pd.concat(t_all, ignore_index=True), pd.DataFrame(p_all), model)
    if not out.empty:
        tmp = out.copy()
        tmp['cost_bps'] = tmp['scenario'].str.replace('cost_', '', regex=False).str.replace('bps', '', regex=False).astype(float)
        be = float(tmp[tmp['sharpe_ratio'] >= 0]['cost_bps'].max()) if not tmp[tmp['sharpe_ratio'] >= 0].empty else 0.0
        out['break_even_cost_bps'] = be
    return out


def _schema(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def _write_reports(class_df: pd.DataFrame, back_df: pd.DataFrame, oos_df: pd.DataFrame, costs_df: pd.DataFrame, manifest: dict, report_out: Path):
    def pct(v): return 'n/a' if pd.isna(v) else f"{100*v:.1f}%"
    def num(v, n=4): return 'n/a' if pd.isna(v) else f"{v:.{n}f}"

    lines = ['# AlphaLens Executive Diagnostic Report', '', f"Generated: {manifest['run_timestamp_utc']}", f"Commit: {manifest['git_commit']}", f"Data mode: {manifest['data_mode']}", '']
    lines += ['## Classification', '| Model | Accuracy | Weighted F1 | Macro F1 |', '|---|---:|---:|---:|']
    for _, r in class_df[class_df['class'] == 'overall'].iterrows():
        lines.append(f"| {r['model']} | {num(r['accuracy'])} | {num(r['weighted_f1'])} | {num(r['macro_f1'])} |")

    lines += ['', '## Baseline Backtest', '| Model | Ann Return | Sharpe | Max DD | Win Rate |', '|---|---:|---:|---:|---:|']
    for _, r in back_df.iterrows():
        lines.append(f"| {r['model']} | {pct(r['annualised_return'])} | {num(r['sharpe_ratio'])} | {pct(r['max_drawdown'])} | {pct(r['win_rate'])} |")

    lines += ['', '## Walk-Forward OOS', '| Model | Scenario | Ann Return | Sharpe | IC | +OOS folds | Break-even bps |', '|---|---|---:|---:|---:|---:|---:|']
    for _, r in oos_df.iterrows():
        lines.append(f"| {r['model']} | {r['scenario']} | {pct(r['annualised_return'])} | {num(r['sharpe_ratio'])} | {num(r['ic'])} | {pct(r['pct_positive_oos_folds'])} | {num(r['break_even_cost_bps'],2)} |")

    lines += ['', '## Cost Robustness', '| Model | Scenario | Sharpe | Ann Return |', '|---|---|---:|---:|']
    for _, r in costs_df.iterrows():
        lines.append(f"| {r['model']} | {r['scenario']} | {num(r['sharpe_ratio'])} | {pct(r['annualised_return'])} |")

    lines += ['', '## Predictive Edge Conclusion']
    if oos_df.empty:
        lines.append('- insufficient OOS evidence')
    else:
        for model in sorted(oos_df['model'].unique()):
            x = oos_df[(oos_df['model'] == model) & (oos_df['scenario'].str.startswith('improved'))]
            if x.empty:
                lines.append(f'- {model}: insufficient improved OOS run')
                continue
            r = x.iloc[0]
            edge = pd.notna(r['sharpe_ratio']) and r['sharpe_ratio'] > 0 and pd.notna(r['ic']) and r['ic'] > 0 and pd.notna(r['pct_positive_oos_folds']) and r['pct_positive_oos_folds'] >= 0.55
            lines.append(f"- {model}: {'evidence of predictive edge' if edge else 'weak/unstable predictive edge'} (Sharpe={num(r['sharpe_ratio'],3)}, IC={num(r['ic'])})")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    (config.METRICS_DIR / 'model_card.md').write_text('\n'.join([
        '# Model Card', '', '## Intended Use', 'Research diagnostics and risk-aware sentiment signal comparison.', '',
        '## Known Limits', '- Free data coverage constraints', '- Classification quality does not imply tradable alpha', '- Strict mode required for claim-making runs', ''
    ]), encoding='utf-8')

    (config.METRICS_DIR / 'strategy_card.md').write_text('\n'.join([
        '# Strategy Card', '', '## Components', '- Sentiment baseline', '- Sentiment + momentum/volatility ensemble', '- Optional regime throttle', '- Liquidity filter and participation cap', '',
        '## Validation', '- Expanding-window quarterly walk-forward', '- Fold-local parameter calibration', '- Cost stress test and break-even bps', ''
    ]), encoding='utf-8')


def run_diagnostics(data_mode='offline_snapshot', strict_data=False, report_out: str = '', build_snapshot=False, capital_usd=1_000_000.0, max_participation=0.02):
    _set_seed(config.RANDOM_SEED)
    states = []

    def add_state(stage, status, code='OK', message=''):
        states.append({'stage': stage, 'status': status, 'code': code, 'message': message, 'timestamp_utc': datetime.now(timezone.utc).isoformat()})

    add_state('run', 'IN_PROGRESS', 'OK', 'Diagnostics started')

    add_state('data_load', 'IN_PROGRESS')
    news, prices, volumes, benchmark, risk_free = _load_data(data_mode=data_mode, strict_data=strict_data, build_snapshot=build_snapshot)
    add_state('data_load', 'SUCCESS')
    coverage = _coverage_ratio(prices)
    if strict_data and coverage < 0.70:
        raise RuntimeError(f'insufficient coverage ratio in strict mode: {coverage:.3f}')

    add_state('model_load', 'IN_PROGRESS')
    finbert_model, finbert_name = _load_finbert_or_fallback()
    add_state('model_load', 'SUCCESS', 'OK', finbert_name)
    models = {'vader': VADERBaseline(), 'finbert': finbert_model if finbert_name == 'finbert' else VADERBaseline()}
    _, _, test_df = load_splits()

    class_rows, back_rows = [], []
    oos_all, period_all, choice_all, cost_all, ablation_all = [], [], [], [], []

    for name, model in models.items():
        cm = evaluate_model(model, test_df, model_name=name)
        class_rows.append({'model': name, 'class': 'overall', 'accuracy': cm['accuracy'], 'weighted_f1': cm['weighted_f1'], 'macro_f1': cm['macro_f1'], 'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
        for c, v in cm['per_class'].items():
            class_rows.append({'model': name, 'class': c, 'accuracy': np.nan, 'weighted_f1': np.nan, 'macro_f1': np.nan, 'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1']})

        scored = _score_news(news, model, name)

        bt = _event_backtest(scored, prices, model_name=name)
        bt['model'] = name
        bt['data_coverage_ratio'] = round(coverage, 4)
        back_rows.append(bt)

        oos, period, choices, ablation = _walk_forward(scored, prices, volumes, name, capital_usd, max_participation)
        if not oos.empty:
            oos['data_coverage_ratio'] = round(coverage, 4)
            oos_all.append(oos)
        if not period.empty: period_all.append(period)
        if not choices.empty: choice_all.append(choices)
        if not ablation.empty: ablation_all.append(ablation)

        costs = _cost_robustness(scored, prices, volumes, name, choices, capital_usd, max_participation)
        if not costs.empty:
            costs['data_coverage_ratio'] = round(coverage, 4)
            cost_all.append(costs)

    class_df = pd.DataFrame(class_rows)
    back_df = pd.DataFrame(back_rows)
    oos_df = pd.concat(oos_all, ignore_index=True) if oos_all else pd.DataFrame()
    period_df = pd.concat(period_all, ignore_index=True) if period_all else pd.DataFrame()
    choice_df = pd.concat(choice_all, ignore_index=True) if choice_all else pd.DataFrame()
    cost_df = pd.concat(cost_all, ignore_index=True) if cost_all else pd.DataFrame()
    ablation_df = pd.concat(ablation_all, ignore_index=True) if ablation_all else pd.DataFrame()

    if not cost_df.empty and not oos_df.empty:
        be = cost_df.groupby('model')['break_even_cost_bps'].max().to_dict()
        oos_df['break_even_cost_bps'] = oos_df['model'].map(be)

    oos_cols = ['model','scenario','annualised_return','sharpe_ratio','max_drawdown','win_rate','trade_count','avg_turnover','ic','ic_stability','oos_fold_count','pct_positive_oos_folds','break_even_cost_bps','data_coverage_ratio']
    back_cols = ['model','model_name','total_return','benchmark_total_return','annualised_return','annualised_volatility','sharpe_ratio','sortino_ratio','calmar_ratio','max_drawdown','win_rate','alpha','beta','total_trades','n_days','ann_turnover','avg_holding_period_days','total_cost_drag','data_coverage_ratio']
    class_cols = ['model','class','accuracy','weighted_f1','macro_f1','precision','recall','f1']

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    add_state('persist', 'IN_PROGRESS')
    _schema(class_df, class_cols).sort_values(['model','class']).to_csv(config.METRICS_DIR / 'diagnostics_classification_metrics.csv', index=False)
    _schema(back_df, back_cols).sort_values(['model']).to_csv(config.METRICS_DIR / 'diagnostics_backtest_metrics.csv', index=False)
    _schema(oos_df, oos_cols).sort_values(['model','scenario']).to_csv(config.METRICS_DIR / 'diagnostics_oos_summary.csv', index=False)
    period_df.to_csv(config.METRICS_DIR / 'diagnostics_oos_period_metrics.csv', index=False)
    choice_df.to_csv(config.METRICS_DIR / 'diagnostics_oos_parameter_choices.csv', index=False)
    _schema(cost_df, oos_cols).sort_values(['model','scenario']).to_csv(config.METRICS_DIR / 'diagnostics_cost_robustness.csv', index=False)
    ablation_df.to_csv(config.METRICS_DIR / 'diagnostics_ablation.csv', index=False)

    ranked = pd.concat([_schema(oos_df, oos_cols), _schema(cost_df, oos_cols)], ignore_index=True).sort_values(['model','sharpe_ratio'], ascending=[True, False])
    ranked.to_csv(config.METRICS_DIR / 'diagnostics_ranked_findings.csv', index=False)

    manifest = {
        'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'config_hash': _hash_file(config.BASE_DIR / 'config.py') if (config.BASE_DIR / 'config.py').exists() else 'unknown',
        'data_hash': _hash_paths([p for p in _snapshot_paths().values() if p.exists()] + [config.PROCESSED_DATA_DIR / 'combined_news.csv']),
        'data_mode': data_mode,
        'strict_data': strict_data,
    }
    (config.METRICS_DIR / 'diagnostics_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pd.DataFrame(states).to_csv(config.METRICS_DIR / 'diagnostics_run_states.csv', index=False)
    add_state('persist', 'SUCCESS')
    pd.DataFrame(states).to_csv(config.METRICS_DIR / 'diagnostics_run_states.csv', index=False)
    add_state('run', 'SUCCESS')
    pd.DataFrame(states).to_csv(config.METRICS_DIR / 'diagnostics_run_states.csv', index=False)

    report_path = Path(report_out) if report_out else (config.METRICS_DIR / 'bank_executive_report.md')
    _write_reports(_schema(class_df, class_cols), _schema(back_df, back_cols), _schema(oos_df, oos_cols), _schema(cost_df, oos_cols), manifest, report_path)

    return {'report_path': str(report_path)}


def main():
    parser = argparse.ArgumentParser(description='Bank-ready AlphaLens diagnostics')
    parser.add_argument('--data-mode', choices=['offline_snapshot', 'live'], default='offline_snapshot')
    parser.add_argument('--strict-data', action='store_true')
    parser.add_argument('--report-out', type=str, default='')
    parser.add_argument('--build-snapshot', action='store_true')
    parser.add_argument('--capital-usd', type=float, default=1_000_000.0)
    parser.add_argument('--max-participation', type=float, default=0.02)
    args = parser.parse_args()

    try:
        result = run_diagnostics(
            data_mode=args.data_mode,
            strict_data=args.strict_data,
            report_out=args.report_out,
            build_snapshot=args.build_snapshot,
            capital_usd=args.capital_usd,
            max_participation=args.max_participation,
        )
    except Exception as exc:
        msg = str(exc)
        code = 'DIAGNOSTICS_ERROR'
        if 'stale' in msg.lower():
            code = 'STALE_CACHE'
        elif 'coverage' in msg.lower():
            code = 'INSUFFICIENT_COVERAGE'
        elif 'missing' in msg.lower() or 'snapshot' in msg.lower() or 'fetch' in msg.lower() or 'data' in msg.lower():
            code = 'DATA_UNAVAILABLE'
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {'stage': 'run', 'status': 'ERROR', 'code': code, 'message': msg, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        ]).to_csv(config.METRICS_DIR / 'diagnostics_run_states.csv', index=False)
        raise

    print('Saved diagnostics outputs to results/metrics/')
    print(f"Executive report: {result['report_path']}")
if __name__ == '__main__':
    main()
