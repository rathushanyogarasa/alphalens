# Research Dataset Pack

Deterministic offline pack used by `run_performance_diagnostics.py --data-mode offline_snapshot`.

## Files
- `news.csv`: normalized headline dataset with `date`, `ticker`, and source fields.
- `prices.csv`: daily close panel (`date` index x ticker columns).
- `volumes.csv`: daily volume panel (`date` index x ticker columns).
- `benchmark.csv`: benchmark close panel (typically `SPY`).
- `risk_free.csv`: daily risk-free proxy series.
- `metadata.json`: pack build timestamp and coverage metadata.

## Coverage window
The active window is defined by `metadata.json` fields:
- `coverage_start`
- `coverage_end`

## Rebuild
Run:

```bash
python run_performance_diagnostics.py --data-mode live --build-snapshot --strict-data
```

This overwrites the pack with the latest validated live dataset.