"""Single-command bank demo wrapper for AlphaLens."""

from run_performance_diagnostics import run_diagnostics


if __name__ == '__main__':
    out = run_diagnostics(
        data_mode='offline_snapshot',
        strict_data=True,
        report_out='',
        build_snapshot=False,
        capital_usd=1_000_000.0,
        max_participation=0.02,
    )
    print('Bank demo complete')
    print(f"Executive report: {out['report_path']}")