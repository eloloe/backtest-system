"""Compatibility import. The implementation is ``backtest_system.fundamentals.pit``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.fundamentals.pit as _impl

sys.modules[__name__] = _impl
