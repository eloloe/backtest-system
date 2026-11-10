"""Command line for the same backtest the Streamlit app runs.

    python cli.py --config examples/trend_stop.yaml
    python -m cli --config examples/cross_section.yaml --start 2016-01-01

Keys stay in the environment (``TIINGO_API_KEY`` and the rest). The yaml
names a provider; it does not carry the key.
"""

from __future__ import annotations

import argparse
import math
import sys

from backtest_system.engine.crosscheck import OptionalEngineError
from backtest_system.portfolio_app.bundle import build_portfolio_bundle
from backtest_system.research.runner import (
    ConfigError,
    ResearchError,
    attach_crosscheck,
    format_metrics,
    load_run_config,
    run_research,
    write_report,
)
from backtest_system.research.walk_forward import WalkForwardError


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
    parser.add_argument("--benchmark", help="基準代碼。省略時依市場用 SPY 或 0050.TW；空字串表示不要")
    parser.add_argument("--provider", help="價格來源，金鑰仍從環境變數讀")
    parser.add_argument("--output", dest="output_dir", help="圖和指標的輸出目錄")
    parser.add_argument("--walk-forward", action="store_true", help="改跑滾動測試窗")
    parser.add_argument("--no-walk-forward", action="store_true", help="即使 YAML 開了也改跑整段")
    parser.add_argument("--open-holdout", action="store_true", help="另外評鎖定的 holdout，不拿它選參")
    parser.add_argument(
        "--cross-check",
        choices=["reference", "nautilus"],
        help="跑完後用第二套帳本對總損益、淨值、成交筆數和成交價",
    )
    parser.add_argument("--mini-app-dir", help="另外輸出可嵌入網站的互動診斷小 app 靜態包")
    parser.add_argument("--mini-app-data-label", default="歷史回測", help="小 app 的資料標籤")
    parser.add_argument("--mini-app-validation", default="樣本內", help="小 app 的驗證標籤")
    parser.add_argument("--mini-app-title", default="Leo｜回測診斷與假設檢查")
    parser.add_argument("--mini-app-subtitle", default="證據導向：先看數字，再下結論")
    parser.add_argument("--shioaji-fixture", help="小 app 可帶入離線 quote fixture JSON（只讀）")
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config, _overrides(args))
        result = run_research(config)
    except (ConfigError, ResearchError, WalkForwardError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report = None
    cross_error = None
    if config.cross_check:
        try:
            report = attach_crosscheck(result)
        except OptionalEngineError as exc:
            cross_error = str(exc)
            result.notes.append(str(exc))
        except (ResearchError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    written = write_report(result, config.output_dir)
    print(format_metrics(result.metrics_display))
    holdout = None if result.walk_forward is None else result.walk_forward.holdout
    if holdout is not None:
        print()
        print("鎖定 holdout")
        print(format_metrics(holdout.metrics_display))
    if result.notes:
        print()
        for note in result.notes:
            print(note)
    print()
    print(f"淨值圖    {written['equity_html']}")
    print(f"回撤圖    {written['drawdown_html']}")
    print(f"滾動夏普  {written['rolling_sharpe_html']}")
    print(f"指標      {written['metrics_json']}")
    print(f"對照      {written['baselines_csv']}")
    if "trials_csv" in written:
        print(f"試驗      {written['trials_csv']}")
    if "crosscheck_json" in written:
        print(f"對帳檔    {written['crosscheck_json']}")
    if args.mini_app_dir:
        mini = build_portfolio_bundle(
            result,
            output_dir=args.mini_app_dir,
            title=args.mini_app_title,
            subtitle=args.mini_app_subtitle,
            data_label=args.mini_app_data_label,
            validation_label=args.mini_app_validation,
            shioaji_fixture=args.shioaji_fixture,
        )
        print(f"小 app    {mini['index_html']}")
        print(f"證據包    {mini['data_json']}")
    if report is not None:
        state = "吻合" if report.ok else "分叉"
        print(f"對帳      {config.cross_check} {state}  最大淨值差 {_gap(report.max_abs_diff)}")
    if cross_error:
        print(cross_error, file=sys.stderr)
        return 1
    if report is not None and not report.ok:
        return 2
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
    if args.cross_check:
        picked["cross_check"] = args.cross_check
    return picked


def _gap(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "inf"
    if not math.isfinite(number):
        return "inf"
    return f"{number:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
