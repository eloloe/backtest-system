"""Compatibility import. The implementation is ``backtest_system.providers.base``."""

from _bootstrap import ensure_src

ensure_src()

import sys

import backtest_system.providers.base as _impl

sys.modules[__name__] = _impl
