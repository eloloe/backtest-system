"""Compatibility import. The implementation is ``backtest_system.engine.backtester``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.engine.backtester as _impl

sys.modules[__name__] = _impl
