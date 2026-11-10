"""CLI for local read-only Shioaji quote capture."""

from __future__ import annotations

import argparse

from backtest_system.portfolio_app.shioaji_capture import capture_quote_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本機 Shioaji 只讀 quote 收集器（輸出 fixture JSON）")
    parser.add_argument("--symbols", required=True, help="逗號分隔，例如 2330.TW,0050.TW,TXF")
    parser.add_argument("--seconds", type=float, default=15.0, help="收集秒數")
    parser.add_argument("--output", required=True, help="輸出 JSON 路徑")
    parser.add_argument("--no-depth", action="store_true", help="不輸出 depth 欄位")
    parser.add_argument("--production", action="store_true", help="覆寫為 simulation=False")
    args = parser.parse_args(argv)

    symbols = [part.strip() for part in args.symbols.split(",") if part.strip()]
    path = capture_quote_fixture(
        symbols=symbols,
        duration_seconds=float(args.seconds),
        output_path=args.output,
        include_depth=not args.no_depth,
        simulation=False if args.production else None,
    )
    print(f"fixture: {path}")
    print("note: this tool subscribes quotes only; no order/trade API calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
