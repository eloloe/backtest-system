"""CLI to export a website-ready interactive diagnosis mini app bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest_system.portfolio_app.bundle import build_portfolio_bundle
from backtest_system.portfolio_app.demo_data import build_demo_dataset
from backtest_system.research.runner import ConfigError, ResearchError, load_run_config, run_research
from backtest_system.research.walk_forward import WalkForwardError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="輸出可嵌入網站的回測診斷小 app 靜態包。")
    parser.add_argument("--config", required=True, help="回測設定檔")
    parser.add_argument("--output", required=True, help="靜態包輸出目錄")
    parser.add_argument("--title", default="Leo｜回測診斷與假設檢查")
    parser.add_argument("--subtitle", default="證據導向：先看數字，再下結論")
    parser.add_argument("--data-label", default="歷史回測", help="例如：歷史回測 / 合成樣本 / 即時觀察")
    parser.add_argument("--validation-label", default="樣本內", help="例如：樣本內 / walk-forward 樣本外")
    parser.add_argument("--demo-synthetic", action="store_true", help="使用內建合成價格，無需 API key")
    parser.add_argument("--shioaji-fixture", help="離線 quote fixture JSON（只讀）")
    parser.add_argument("--shioaji-slippage-bps", type=float, default=5.0)
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config)
        if args.demo_synthetic:
            demo = build_demo_dataset()
            result = run_research(config, prices=demo.prices, benchmark_close=demo.benchmark_close)
        else:
            result = run_research(config)
    except (ConfigError, ResearchError, WalkForwardError, ValueError) as exc:
        print(str(exc))
        return 1

    written = build_portfolio_bundle(
        result,
        output_dir=args.output,
        title=args.title,
        subtitle=args.subtitle,
        data_label=args.data_label,
        validation_label=args.validation_label,
        shioaji_fixture=args.shioaji_fixture,
        shioaji_slippage_bps=float(args.shioaji_slippage_bps),
    )
    print(f"mini app: {written['index_html']}")
    print(f"evidence: {written['data_json']}")
    print(f"integration: {written['integration_readme']}")
    summary_path = Path(args.output) / "bundle_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "index_html": str(written["index_html"]),
                "data_json": str(written["data_json"]),
                "integration_readme": str(written["integration_readme"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
