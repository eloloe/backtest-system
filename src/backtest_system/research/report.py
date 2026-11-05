"""Charts and files shared by the CLI and the Streamlit app.

One finished run writes an equity curve (strategy, buy-and-hold benchmark,
noise path), a drawdown, a 252-day rolling Sharpe, and the metrics table.
``metrics.json`` also carries the structured selection block: trial count,
PSR, and, when more than one parameter set was scored, the deflated Sharpe
and the Holm decisions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from backtest_system.research.walk_forward import equity_figure

_COLOR_BENCH = "#F0883E"
_COLOR_NOISE = "#8B949E"
_COLOR_BUY = "#3FB950"
_COLOR_SELL = "#F85149"


def comparison_equity_figure(equity: pd.Series, title: str, extras: Sequence[tuple] = (), trades=None) -> go.Figure:
    """Strategy equity, then any benchmark or noise curves, then trade markers."""
    fig = equity_figure(equity, title)
    for name, series, dash, color in extras:
        if series is None or len(series) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=str(name),
                line=dict(color=color, width=2, dash=dash),
            )
        )
    if trades is not None and not getattr(trades, "empty", True) and "action" in trades.columns:
        _add_trades(fig, equity, trades)
    return fig


def rolling_sharpe_figure(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 252,
    title: str = "滾動夏普（252 天）",
) -> go.Figure:
    """Same rolling Sharpe the single-run page used to draw inline."""
    clean = pd.Series(returns, dtype="float64").replace([float("inf"), float("-inf")], pd.NA)
    rolling = clean.rolling(int(window)).apply(
        lambda sample: _window_sharpe(sample, risk_free_rate),
        raw=False,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling,
            name="滾動夏普",
            line=dict(color="#BC8CFF", width=2),
        )
    )
    fig.add_hline(y=1, line_dash="dot", line_color=_COLOR_BUY, annotation_text="1")
    fig.add_hline(y=0, line_dash="dot", line_color="#8B949E")
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="夏普比率",
        template="plotly_dark",
        height=280,
        margin=dict(t=50, b=40),
    )
    return fig


def write_report(result, output_dir) -> dict[str, Path]:
    """Write the standard pack. Paths are absolute ``Path`` objects."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    equity_path = directory / "equity.html"
    drawdown_path = directory / "drawdown.html"
    rolling_path = directory / "rolling_sharpe.html"
    result.equity_figure.write_html(str(equity_path), include_plotlyjs="cdn")
    result.drawdown_figure.write_html(str(drawdown_path), include_plotlyjs="cdn")
    rolling = result.rolling_sharpe_figure
    if rolling is not None:
        rolling.write_html(str(rolling_path), include_plotlyjs="cdn")
    result.equity.rename("equity").to_csv(directory / "equity.csv")

    display_frame = pd.DataFrame(list(result.metrics_display.items()), columns=["指標", "數值"])
    display_frame.to_csv(directory / "metrics.csv", index=False)
    payload = {
        "strategy": result.config.strategy,
        "symbols": result.symbols,
        "start": result.config.start,
        "end": result.config.end,
        "walk_forward": result.config.walk_forward,
        "benchmark": result.config.benchmark,
        "metrics": result.metrics_display,
        "metrics_raw": _jsonable(result.metrics),
        "comparison": _jsonable(getattr(result, "comparison", {}) or {}),
        "selection": _jsonable(getattr(result, "selection", None)),
        "notes": result.notes,
    }
    walk = result.walk_forward
    if walk is not None and walk.holdout is not None:
        payload["holdout_selection"] = _jsonable(getattr(walk.holdout, "selection", None))
        payload["holdout_metrics"] = walk.holdout.metrics_display
    (directory / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written = {
        "equity_html": equity_path,
        "drawdown_html": drawdown_path,
        "rolling_sharpe_html": rolling_path,
        "equity_csv": directory / "equity.csv",
        "metrics_csv": directory / "metrics.csv",
        "metrics_json": directory / "metrics.json",
    }
    baselines = baseline_frame(result)
    baselines.to_csv(directory / "baselines.csv", index=False)
    written["baselines_csv"] = directory / "baselines.csv"
    if result.trades is not None and not result.trades.empty:
        result.trades.to_csv(directory / "trades.csv", index=False)
        written["trades_csv"] = directory / "trades.csv"
    if result.holdings is not None and not result.holdings.empty:
        result.holdings.to_csv(directory / "holdings.csv", index=False)
        written["holdings_csv"] = directory / "holdings.csv"
    if walk is not None:
        fold_table(walk).to_csv(directory / "folds.csv", index=False)
        written["folds_csv"] = directory / "folds.csv"
        frame = trials_frame(getattr(walk, "trials", []), getattr(result, "selection", None))
        if frame is not None and not frame.empty:
            frame.to_csv(directory / "trials.csv", index=False)
            written["trials_csv"] = directory / "trials.csv"
        if walk.holdout is not None:
            holdout = walk.holdout
            holdout.equity_figure.write_html(
                str(directory / "holdout_equity.html"), include_plotlyjs="cdn"
            )
            holdout.drawdown_figure.write_html(
                str(directory / "holdout_drawdown.html"), include_plotlyjs="cdn"
            )
            written["holdout_equity_html"] = directory / "holdout_equity.html"
            written["holdout_drawdown_html"] = directory / "holdout_drawdown.html"
    return written


def baseline_frame(result) -> pd.DataFrame:
    rows = [
        {
            "name": "strategy",
            "sharpe": result.metrics.get("sharpe"),
            "total_return": result.metrics.get("total_return"),
            "max_drawdown": result.metrics.get("max_drawdown"),
        }
    ]
    comparison = getattr(result, "comparison", {}) or {}
    benchmark = comparison.get("benchmark") or {}
    if benchmark:
        rows.append(
            {
                "name": f"benchmark:{benchmark.get('symbol') or ''}",
                "sharpe": benchmark.get("sharpe"),
                "total_return": benchmark.get("total_return"),
                "max_drawdown": benchmark.get("max_drawdown"),
            }
        )
    noise = comparison.get("noise") or {}
    if noise:
        rows.append(
            {
                "name": noise.get("kind") or "noise",
                "sharpe": noise.get("sharpe_median"),
                "total_return": noise.get("total_return_median"),
                "max_drawdown": noise.get("max_drawdown_median"),
            }
        )
    return pd.DataFrame(rows)


def trials_frame(trials: Sequence, selection: Mapping | None) -> pd.DataFrame:
    if not trials:
        return pd.DataFrame()
    holm = _by_label(selection, "holm")
    bh = _by_label(selection, "bh")
    rows = []
    for trial in trials:
        label = trial.label
        left = holm.get(label, {})
        right = bh.get(label, {})
        rows.append(
            {
                "label": label,
                "fold": trial.fold_id + 1,
                "params": json.dumps(trial.params, ensure_ascii=False, default=str),
                "sharpe": trial.sharpe,
                "p_value": trial.p_value,
                "holm_rank": left.get("rank"),
                "holm_threshold": left.get("threshold"),
                "holm_adjusted_p": left.get("adjusted_p"),
                "holm_reject": left.get("reject"),
                "bh_adjusted_p": right.get("adjusted_p"),
                "bh_reject": right.get("reject"),
            }
        )
    return pd.DataFrame(rows)


def fold_table(result) -> pd.DataFrame:
    rows = []
    for fold in result.folds:
        rows.append(
            {
                "fold": fold.fold_id + 1,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "as_of": fold.as_of.date().isoformat(),
                "symbols": ", ".join(fold.symbols),
                "params": "" if not fold.params else json.dumps(fold.params, ensure_ascii=False, default=str),
                "train_score": fold.train_score,
                "test_sharpe": fold.test_sharpe,
                "test_total_return": fold.test_total_return,
                "note": fold.note,
            }
        )
    return pd.DataFrame(rows)


def comparison_lines(comparison: Mapping) -> dict[str, str]:
    lines = {}
    benchmark = comparison.get("benchmark") or {}
    sharpe = _finite(benchmark.get("sharpe"))
    if sharpe is not None:
        symbol = benchmark.get("symbol") or "基準"
        lines["基準買進持有 Sharpe"] = f"{sharpe:.3f}（{symbol}，不含成本）"
    noise = comparison.get("noise") or {}
    median = _finite(noise.get("sharpe_median"))
    if median is not None:
        kind = "隨機進場 Sharpe" if noise.get("kind") == "random_entry" else "打亂標籤 Sharpe"
        if noise.get("degenerate"):
            lines[kind] = f"{median:.3f}（訊號沒有變化）"
        else:
            lines[kind] = f"{median:.3f}（{int(noise.get('n_perm') or 0)} 次的中位數）"
    return lines


def _add_trades(fig: go.Figure, equity: pd.Series, trades: pd.DataFrame) -> None:
    dated = trades.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    for action, name, color, symbol in (
        ("BUY", "買入", _COLOR_BUY, "triangle-up"),
        ("SELL", "賣出", _COLOR_SELL, "triangle-down"),
    ):
        side = dated[dated["action"] == action]
        if side.empty:
            continue
        marks = equity.reindex(pd.DatetimeIndex(side["date"])).dropna()
        if marks.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=marks.index,
                y=marks,
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, symbol=symbol),
            )
        )


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _window_sharpe(sample: pd.Series, risk_free_rate: float) -> float:
    std = float(sample.std())
    if not math.isfinite(std) or std <= 0:
        return 0.0
    return (float(sample.mean()) * 252 - float(risk_free_rate)) / (std * math.sqrt(252))


def _by_label(selection: Mapping | None, key: str) -> dict:
    if not selection:
        return {}
    block = selection.get(key) or {}
    return {row["label"]: row for row in block.get("rows") or []}


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            pass
    return value
