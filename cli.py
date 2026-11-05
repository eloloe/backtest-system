"""Command line for the same backtest the Streamlit app runs.

    python cli.py --config examples/trend_stop.yaml
    python -m cli --config examples/cross_section.yaml --start 2016-01-01

Keys stay in the environment (``TIINGO_API_KEY`` and the rest). The yaml
names a provider; it does not carry the key.
"""

from __future__ import annotations

import argparse
import sys

from runner import ConfigError, ResearchError, format_metrics, load_run_config, run_research, write_report
from walk_forward import WalkForwardError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="跑和 Streamlit 同一條回測，寫出淨值、回撤和指標。"
    )
    parser.add_argument("--config", required=True, help="YAML 設定檔")
    parser.add_argument("--strategy", help="蓋過 YAML 的策略名稱")
    parser.add_argument("--tickers", help="逗號分隔，蓋過 YAML 的商品池")
    parser.add_argument("--start", help="起始日，含當天")
    parser.add_argument("--end", help="結束日，不含當天")
    parser.add_argument("--capital", type=float, dest="initial_capital", help="初始資金")
    parser.add_argument("--benchmark", help="基準代碼，空字串表示不要")
    parser.add_argument("--provider", help="價格來源，金鑰仍從環境變數讀")
    parser.add_argument("--output", dest="output_dir", help="圖和指標的輸出目錄")
    parser.add_argument("--walk-forward", action="store_true", help="改跑滾動測試窗")
    parser.add_argument("--no-walk-forward", action="store_true", help="即使 YAML 開了也改跑整段")
    parser.add_argument("--open-holdout", action="store_true", help="另外評鎖定的 holdout，不拿它選參")
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config, _overrides(args))
        result = run_research(config)
    except (ConfigError, ResearchError, WalkForwardError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    written = write_report(result, config.output_dir)
    print(format_metrics(result.metrics_display))
    if result.notes:
        print()
        for note in result.notes:
            print(note)
    print()
    print(f"淨值圖  {written['equity_html']}")
    print(f"回撤圖  {written['drawdown_html']}")
    print(f"指標    {written['metrics_json']}")
    return 0


def _overrides(args: argparse.Namespace) -> dict:
    if args.walk_forward and args.no_walk_forward:
        raise ConfigError("不要同時加 --walk-forward 和 --no-walk-forward")
    picked = {}
    for name in (
        "strategy",
        "tickers",
        "start",
        "end",
        "initial_capital",
        "benchmark",
        "provider",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            picked[name] = value
    if args.walk_forward:
        picked["walk_forward"] = True
    if args.no_walk_forward:
        picked["walk_forward"] = False
    if args.open_holdout:
        picked["open_holdout"] = True
    return picked


if __name__ == "__main__":
    raise SystemExit(main())
