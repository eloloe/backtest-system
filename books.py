"""Compatibility import. The implementation is ``backtest_system.engine.portfolio``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.engine.portfolio as _impl

sys.modules[__name__] = _impl
