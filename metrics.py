"""Compatibility import. The implementation is ``backtest_system.research.metrics``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.research.metrics as _impl

sys.modules[__name__] = _impl
