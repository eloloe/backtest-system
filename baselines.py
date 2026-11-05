"""Compatibility import. The implementation is ``backtest_system.research.baselines``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.research.baselines as _impl

sys.modules[__name__] = _impl
