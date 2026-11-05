"""Compatibility import. The implementation is ``backtest_system.research.runner``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.research.runner as _impl

sys.modules[__name__] = _impl
