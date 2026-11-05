"""Compatibility import. The implementation is ``backtest_system.providers.shioaji``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.providers.shioaji as _impl

sys.modules[__name__] = _impl
