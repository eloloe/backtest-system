"""Compatibility import. The implementation is ``backtest_system.strategies``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.strategies as _impl

sys.modules[__name__] = _impl
