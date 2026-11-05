"""Compatibility import. The implementation is ``backtest_system.fundamentals.screener``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.fundamentals.screener as _impl

sys.modules[__name__] = _impl
