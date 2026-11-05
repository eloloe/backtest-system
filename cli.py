"""Command line for the same backtest the Streamlit app runs.

    python cli.py --config examples/trend_stop.yaml
    python -m cli --config examples/cross_section.yaml --start 2016-01-01

The installed console script is ``backtest``. Keys stay in the environment.
"""

from _bootstrap import ensure_src

ensure_src()

from backtest_system.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
