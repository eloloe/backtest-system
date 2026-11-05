
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from baselines import infer_benchmark
from fundamentals import fetch_fundamentals, fetch_fundamentals_pit, reporting_lag_days
from factor_engine import FACTOR_REGISTRY
from screener import apply_factor_score, filter_stocks
from strategies import canonical_strategy
from runner import ResearchError, RunConfig, run_research
from walk_forward import WalkForwardError

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="金融回測系統",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ──────────────────────────────────────────────
# 自訂 CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* 全域背景 */
.stApp { background-color: #0D1117; color: #E6EDF3; }

/* 側邊欄 */
section[data-testid="stSidebar"] { background-color: #161B22; }

/* 側邊欄文字（標題、標籤、說明）強制為亮色，避免深色底配深灰字看不到 */
section[data-testid="stSidebar"] * { color: #E6EDF3; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F0F6FC; }

/* 一般文字、Widget 標籤 */
.stApp, .stApp p, .stApp label, .stApp span,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p { color: #E6EDF3; }

/* 輸入框 / 數字框內文字保持深底亮字 */
.stApp input, .stApp textarea { color: #E6EDF3 !important; }

/* 說明用的 st.info / st.error 等提示區塊文字 */
[data-testid="stAlert"] * { color: #E6EDF3; }

/* 標題區塊 */
.hero {
    background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 2rem; color: #E6EDF3; }
.hero p  { margin: 6px 0 0; color: #8B949E; font-size: 0.95rem; }

/* KPI 卡片 */
.kpi-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi-card {
    flex: 1; min-width: 130px;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
}
.kpi-label { font-size: 0.75rem; color: #8B949E; margin-bottom: 6px; }
.kpi-value { font-size: 1.6rem; font-weight: 700; }
.kpi-sub   { font-size: 0.75rem; color: #8B949E; margin-top: 4px; }
.pos { color: #3FB950; }
.neg { color: #F85149; }
.neu { color: #58A6FF; }

/* 指標表格美化 */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* 分隔線 */
hr { border-color: #30363D; }
</style>
""",
    unsafe_allow_html=True,
)

STRATEGY_LIST = [
    "買進持有 (Buy & Hold)",
    "均線交叉 (MA Cross)",
    "RSI 策略",
    "布林通道 (Bollinger Bands)",
    "MACD 策略",
    "趨勢停損 (Trend + Stop)",
    "橫截面多因子 (Cross-Section)",
    "核心衛星 (Core-Satellite)",
]

PLOTLY_THEME = "plotly_dark"
COLOR_STRATEGY = "#58A6FF"
COLOR_BENCHMARK = "#F0883E"
COLOR_BUY = "#3FB950"
COLOR_SELL = "#F85149"
COLOR_DD = "rgba(248, 81, 73, 0.25)"


def sign_class(v: float, positive_good: bool = True) -> str:
    if v > 0:
        return "pos" if positive_good else "neg"
    if v < 0:
        return "neg" if positive_good else "pos"
    return "neu"


def kpi_card(label: str, value: str, sub: str = "", css_class: str = "neu") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {css_class}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def _finite_text(value, pattern: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number or number in (float("inf"), float("-inf")):
        return ""
    return pattern.format(number)


def _fmt_params(params) -> str:
    if params is None:
        return "沒有選到"
    if not params:
        return "不搜尋"
    return ", ".join(f"{key}={params[key]}" for key in sorted(params))


def _draw_signal_chart(data, plot_trades) -> None:
    fig_sig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )
    fig_sig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"],
            name="收盤價", line=dict(color=COLOR_STRATEGY, width=1.5),
        ),
        row=1, col=1,
    )
    buys = plot_trades[plot_trades["action"] == "BUY"] if not plot_trades.empty else plot_trades
    sells = plot_trades[plot_trades["action"] == "SELL"] if not plot_trades.empty else plot_trades
    if not buys.empty:
        fig_sig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=data["Close"].reindex(buys["date"]).values,
                mode="markers", name="買入",
                marker=dict(color=COLOR_BUY, size=12, symbol="triangle-up"),
            ),
            row=1, col=1,
        )
    if not sells.empty:
        fig_sig.add_trace(
            go.Scatter(
                x=sells["date"],
                y=data["Close"].reindex(sells["date"]).values,
                mode="markers", name="賣出",
                marker=dict(color=COLOR_SELL, size=12, symbol="triangle-down"),
            ),
            row=1, col=1,
        )
    if "Volume" in data.columns:
        fig_sig.add_trace(
            go.Bar(
                x=data.index, y=data["Volume"],
                name="成交量", marker_color="rgba(88,166,255,0.3)",
            ),
            row=2, col=1,
        )
    fig_sig.update_layout(
        template=PLOTLY_THEME, height=520,
        legend=dict(orientation="h", y=1.04),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_sig, use_container_width=True)


def _benchmark_symbol(text, tickers) -> str:
    cleaned = str(text or "").strip()
    if cleaned:
        return cleaned.upper()
    return infer_benchmark(tickers)


def _render_rigor(research, *, expanded: bool) -> None:
    """Benchmark, noise, and — once a grid was searched — DSR and Holm."""
    comparison = getattr(research, "comparison", None) or {}
    benchmark = comparison.get("benchmark") or {}
    noise = comparison.get("noise") or {}
    selection = getattr(research, "selection", None) or {}
    st.markdown("#### 對照")
    left, middle, right = st.columns(3)
    left.metric("策略 Sharpe", f"{float(research.metrics.get('sharpe') or 0):.2f}")
    bench_sharpe = benchmark.get("sharpe")
    middle.metric(
        f"基準 {benchmark.get('symbol') or '—'}",
        "—" if bench_sharpe is None else f"{float(bench_sharpe):.2f}",
    )
    noise_sharpe = noise.get("sharpe_median")
    noise_label = "隨機進場" if noise.get("kind") == "random_entry" else "打亂標籤"
    right.metric(noise_label, "—" if noise_sharpe is None else f"{float(noise_sharpe):.2f}")
    if selection.get("multiple_testing"):
        try:
            dsr_text = f"{float(selection.get('dsr')):.3f}"
        except (TypeError, ValueError):
            dsr_text = "算不出"
        holm = selection.get("holm") or {}
        st.caption(
            f"Deflated Sharpe {dsr_text}，N={selection.get('n_label')}。"
            f"Holm 存活 {holm.get('n_rejected', 0)} / {holm.get('n', 0)}。"
        )
        rows = holm.get("rows") or []
        if rows:
            table = pd.DataFrame(rows)[
                ["label", "statistic", "p_value", "threshold", "adjusted_p", "reject"]
            ].rename(
                columns={
                    "label": "變體",
                    "statistic": "訓練 Sharpe",
                    "p_value": "p",
                    "threshold": "Holm 門檻",
                    "adjusted_p": "調整後 p",
                    "reject": "存活",
                }
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
    elif expanded:
        st.caption("這次沒有在參數之間做選擇，N=1，不計算 Deflated Sharpe。")


def _kpi_html(metrics_raw: dict) -> str:
    tr = metrics_raw["total_return"]
    sh = metrics_raw["sharpe"]
    md = metrics_raw["max_drawdown"]
    cg = metrics_raw["cagr"]
    wr = metrics_raw["win_rate"]
    vl = metrics_raw["annual_vol"]
    return (
        '<div class="kpi-grid">'
        + kpi_card("總報酬率", f"{tr:.2%}", f"CAGR {cg:.2%}", sign_class(tr))
        + kpi_card("夏普比率", f"{sh:.2f}", ">1 佳  >2 優秀", sign_class(sh))
        + kpi_card("最大回撤", f"{md:.2%}", f"Calmar {metrics_raw['calmar']:.2f}", sign_class(md, False))
        + kpi_card("年化報酬", f"{cg:.2%}", "", sign_class(cg))
        + kpi_card("勝率", f"{wr:.1%}", f"獲利/虧損 {metrics_raw['win_loss_ratio']:.2f}x", sign_class(wr - 0.5))
        + kpi_card("年化波動", f"{vl:.2%}", f"VaR95% {metrics_raw['var_95']:.2%}", "neg" if vl > 0.25 else "neu")
        + "</div>"
    )


def render_walk_forward(
    *,
    universe,
    benchmark,
    start_date,
    end_date,
    initial_capital,
    commission,
    slippage,
    risk_free_rate,
    strategy_name,
    train_months,
    test_months,
    gap_days,
    holdout_months,
    open_holdout,
    use_live_snapshot,
    use_screener,
    pe_max,
    roe_min,
    rev_growth_min,
    mcap_min_b,
    use_factor_scoring,
    active_factors,
    factor_top_n,
    norm_method_key,
):
    """Rolling train / embargo / test. The single full-sample path is not drawn."""
    if use_live_snapshot:
        st.warning(
            "Walk-forward 不使用 Yahoo 今天的快照。"
            "有開篩選或因子時，每一折改以訓練窗第一天已公開的財報選股。"
        )

    mcap = mcap_min_b * 1_000_000_000 if use_screener and mcap_min_b else None
    config = RunConfig(
        strategy=strategy_name,
        tickers=list(universe),
        start=str(pd.Timestamp(start_date).date()),
        end=str(pd.Timestamp(end_date).date()),
        benchmark=_benchmark_symbol(benchmark, universe),
        initial_capital=float(initial_capital),
        commission=float(commission),
        slippage=float(slippage),
        risk_free_rate=float(risk_free_rate),
        walk_forward=True,
        train_months=int(train_months),
        test_months=int(test_months),
        gap_days=int(gap_days),
        holdout_months=int(holdout_months),
        open_holdout=bool(open_holdout),
        params=dict(strategy_params),
        cross_section=dict(cross_section_params),
        core_satellite=dict(core_satellite_params),
        use_screener=bool(use_screener),
        use_factor_scoring=bool(use_factor_scoring and active_factors),
        pe_max=pe_max if use_screener and pe_max and pe_max > 0 else None,
        roe_min=roe_min if use_screener and roe_min is not None and roe_min > -10 else None,
        rev_growth_min=(
            rev_growth_min
            if use_screener and rev_growth_min is not None and rev_growth_min > -10
            else None
        ),
        mcap_min=mcap,
        active_factors=dict(active_factors) if use_factor_scoring else {},
        factor_top_n=int(factor_top_n) if use_factor_scoring else None,
        norm_method=norm_method_key,
    )
    try:
        with st.spinner("下載報價，準備 walk-forward…"):
            research = run_research(config)
    except WalkForwardError as exc:
        st.error(f"Walk-forward 排不下去：{exc}")
        st.stop()
    except ResearchError as exc:
        st.error(str(exc))
        st.stop()
    for note in research.notes:
        st.caption(note)
    result = research.walk_forward

    st.markdown(f"### Walk-forward　｜　{strategy_name}")
    research_end = result.holdout_start or pd.Timestamp(end_date)
    if canonical_strategy(strategy_name) == "cross_section":
        selection_note = "橫截面在測試窗的再平衡日自己排名，不用訓練窗第一天的名單。"
    else:
        selection_note = "選股日是訓練窗的第一天。"
    st.caption(
        f"研究區間 {pd.Timestamp(start_date):%Y-%m-%d} 到 {research_end:%Y-%m-%d}（不含結束）。"
        f"每一折是訓練 {int(train_months)} 個月、禁運 {int(gap_days)} 個日曆日、測試 {int(test_months)} 個月。"
        "選參只用該折訓練窗的價格；測試窗彼此不重疊。迄日都不含當天。"
        + selection_note
    )
    st.markdown(_kpi_html(result.metrics), unsafe_allow_html=True)
    _render_rigor(research, expanded=True)
    st.plotly_chart(research.equity_figure, use_container_width=True)
    st.plotly_chart(research.drawdown_figure, use_container_width=True)
    st.plotly_chart(research.rolling_sharpe_figure, use_container_width=True)
    st.caption("淨值只接各折的測試窗。訓練、禁運，以及沒打開的 holdout，都不在這兩張圖上。每一折從現金重新開始。基準與噪音對照畫在同一張淨值上。")

    rows = []
    for fold in result.folds:
        rows.append(
            {
                "折": fold.fold_id + 1,
                "訓練起": fold.train_start.date().isoformat(),
                "訓練迄": fold.train_end.date().isoformat(),
                "測試起": fold.test_start.date().isoformat(),
                "測試迄": fold.test_end.date().isoformat(),
                "選股日": fold.as_of.date().isoformat(),
                "標的": ", ".join(fold.symbols) if fold.symbols else "無",
                "參數": _fmt_params(fold.params),
                "訓練分數": _finite_text(fold.train_score, "{:.3f}"),
                "測試 Sharpe": _finite_text(fold.test_sharpe, "{:.3f}"),
                "測試總報酬": _finite_text(fold.test_total_return, "{:.2%}"),
                "備註": fold.note,
            }
        )
    fold_table = pd.DataFrame(rows)
    st.markdown("#### 各折")
    st.dataframe(fold_table, use_container_width=True, hide_index=True)
    st.download_button(
        "下載各折",
        fold_table.to_csv(index=False).encode("utf-8-sig"),
        "walk_forward_folds.csv",
        "text/csv",
    )

    with st.expander("樣本外指標", expanded=False):
        st.dataframe(
            pd.DataFrame(list(result.metrics_display.items()), columns=["指標", "數值"]),
            use_container_width=True,
            hide_index=True,
        )

    if result.holdout_start is not None and result.holdout is None:
        st.info(
            f"末端 {result.holdout_start:%Y-%m-%d} 到 {result.holdout_end:%Y-%m-%d}（不含）"
            "留作鎖定 holdout，這次沒有打開。打開只會另表報告，不會把它併進上面的樣本外曲線，"
            "也不會再用它選參。不要拿那段結果回頭改格點或視窗。"
        )
    if result.holdout is not None:
        st.markdown("#### 鎖定 holdout")
        st.warning(
            "這一段沒有參與選參，用的是最後一折已經定下的參數。"
            "把它拿去改策略，這段就不再是留白。"
        )
        st.caption(
            f"{result.holdout.start:%Y-%m-%d} 到 {result.holdout.end:%Y-%m-%d}（不含）。"
            f"選股日改為 holdout 起點。參數：{_fmt_params(result.holdout.params)}。"
            f"標的：{', '.join(result.holdout.symbols) if result.holdout.symbols else '無'}。"
        )
        st.markdown(_kpi_html(result.holdout.metrics), unsafe_allow_html=True)
        holdout_selection = getattr(result.holdout, "selection", None) or {}
        if holdout_selection.get("multiple_testing"):
            st.caption(holdout_selection.get("note") or "")
            holdout_bits = holdout_selection.get("display") or {}
            if holdout_bits:
                st.dataframe(
                    pd.DataFrame(list(holdout_bits.items()), columns=["指標", "數值"]),
                    use_container_width=True,
                    hide_index=True,
                )
        elif holdout_selection:
            st.caption("Holdout 沿用的試驗次數是 1，不計算 Deflated Sharpe。")
        st.plotly_chart(result.holdout.equity_figure, use_container_width=True)
        st.plotly_chart(result.holdout.drawdown_figure, use_container_width=True)


# ──────────────────────────────────────────────
# 側邊欄
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 回測設定")
    st.markdown("---")

    st.markdown("### 商品池設定 (Stock Universe)")
    universe_str = st.text_area(
        "交易標的 (用逗號分隔)",
        value="AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA",
        help="例如：AAPL, MSFT, SPY",
    )
    universe_list = [s.strip().upper() for s in universe_str.split(",") if s.strip()]
    benchmark = st.text_input(
        "基準指數",
        value="SPY",
        help="買進持有這檔，畫在淨值旁邊。留空則台股用 0050.TW，其餘用 SPY。",
    )

    st.markdown("### 🎯 基本面濾網 (Fundamentals)")
    st.caption(
        "歷史回測用起始日當時已公開的財報，不用今天的快照。"
        "免費的 Yahoo 季報大約只有近幾季，更早的 ROE、營收成長常常是空值。"
    )
    use_live_snapshot = st.checkbox(
        "改用 Yahoo 今天的快照（有前視）",
        value=False,
        help="勾選後篩選用的是現在的 PE、ROE。回測當天還看不到這些數字，不能當成歷史結果。",
    )
    use_screener = st.checkbox("啟用基本面篩選", value=True)
    if use_screener:
        pe_max = st.number_input("PE (本益比) <", value=50.0)
        roe_min = st.number_input("ROE (股東權益報酬率) > (%)", value=15.0) / 100.0
        rev_growth_min = st.number_input("營收成長率 > (%)", value=10.0) / 100.0
        mcap_min_b = st.number_input("市值最小值 (Billion USD)", value=10.0)
    else:
        pe_max = roe_min = rev_growth_min = mcap_min_b = None

    # ── Factor Scoring System ──────────────────────────────
    st.markdown("### 🏆 因子評分系統 (Factor Scoring)")
    use_factor_scoring = st.checkbox("啟用因子評分（選出 Top N 股票）", value=False)
    if use_factor_scoring:
        with st.expander("選擇因子與權重", expanded=True):
            active_factors = {}
            for key, meta in FACTOR_REGISTRY.items():
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    enabled = st.checkbox(meta["label"], value=(key in {"roe", "pe"}), key=f"factor_{key}")
                with col_b:
                    w = st.slider(
                        "權重",
                        min_value=0.0, max_value=3.0, value=1.0, step=0.1,
                        key=f"weight_{key}",
                        disabled=not enabled,
                        label_visibility="collapsed",
                    )
                if enabled:
                    active_factors[key] = w

        col_n, col_m = st.columns(2)
        with col_n:
            factor_top_n = st.number_input("挑選前 N 名", min_value=1, max_value=len(universe_list), value=min(5, len(universe_list)))
        with col_m:
            norm_method = st.radio("標準化方式", ["Z-Score", "百分位排名"], horizontal=True)
        norm_method_key = "zscore" if norm_method == "Z-Score" else "rank"
    else:
        active_factors = {}
        factor_top_n = len(universe_list)
        norm_method_key = "zscore"

    st.markdown("### 回測期間")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("開始日期", value=datetime(2018, 1, 1))
    with col_b:
        end_date = st.date_input("結束日期", value=datetime.today())

    st.markdown("### 資金設定")
    initial_capital = st.number_input(
        "初始資金 (USD)", value=100_000, min_value=1_000, step=10_000
    )
    commission = (
        st.slider(
            "手續費 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="只用於美股與其他非台股代碼，買賣都依成交金額收取。台股（.TW / .TWO）改用券商手續費與賣出證交稅。",
        )
        / 100
    )
    slippage = (
        st.slider(
            "滑價 (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="下一根開盤價往不利方向偏移，買入與賣出都適用。",
        )
        / 100
    )
    st.caption(
        "訊號在當根收盤決定，成交在下一根開盤，不會用同一根收盤價成交。"
        "台股預設券商手續費 0.1425%（最低 20 元）；賣出證交稅為股票 0.3%，"
        "代碼以 00 開頭的 ETF 為 0.1%。"
    )
    risk_free_rate = (
        st.slider("無風險利率 (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100
    )

    st.markdown("### 交易策略")
    strategy_name = st.selectbox("選擇策略", STRATEGY_LIST)
    strategy_params = {}
    cross_section_params = {}
    core_satellite_params = {}

    # 策略參數。整段回測用這裡的數字；Walk-forward 改走內建小格點。
    if strategy_name == "均線交叉 (MA Cross)":
        ma_type = st.radio("均線類型", ["SMA", "EMA"], horizontal=True)
        short_window = st.slider("短期均線", 5, 100, 20)
        long_window = st.slider("長期均線", 20, 300, 60)
        strategy_params = {
            "short_window": int(short_window),
            "long_window": int(long_window),
            "ma_type": ma_type,
        }
    elif strategy_name == "RSI 策略":
        rsi_period = st.slider("RSI 週期", 2, 50, 14)
        rsi_oversold = st.slider("超賣門檻", 10, 45, 30)
        rsi_overbought = st.slider("超買門檻", 55, 90, 70)
        strategy_params = {
            "period": int(rsi_period),
            "oversold": int(rsi_oversold),
            "overbought": int(rsi_overbought),
        }
    elif strategy_name == "布林通道 (Bollinger Bands)":
        bb_period = st.slider("週期", 5, 50, 20)
        bb_std = st.slider("標準差倍數", 1.0, 3.5, 2.0, step=0.1)
        strategy_params = {"period": int(bb_period), "std_dev": float(bb_std)}
    elif strategy_name == "MACD 策略":
        macd_fast = st.slider("快線週期", 5, 30, 12)
        macd_slow = st.slider("慢線週期", 15, 60, 26)
        macd_signal_p = st.slider("信號線週期", 3, 20, 9)
        strategy_params = {
            "fast": int(macd_fast),
            "slow": int(macd_slow),
            "signal_period": int(macd_signal_p),
        }
    elif strategy_name == "趨勢停損 (Trend + Stop)":
        trend_window = st.slider("趨勢均線", 20, 250, 100)
        stop_mode_label = st.radio("停損", ["ATR", "百分比", "較緊者"], horizontal=True)
        stop_mode = {"ATR": "atr", "百分比": "percent", "較緊者": "tighter"}[stop_mode_label]
        atr_mult = st.slider("ATR 倍數", 1.0, 5.0, 3.0, step=0.5)
        stop_pct = st.slider("百分比停損", 0.03, 0.25, 0.08, step=0.01)
        strategy_params = {
            "trend_window": int(trend_window),
            "ma_type": "SMA",
            "stop_mode": stop_mode,
            "atr_period": 14,
            "atr_mult": float(atr_mult),
            "stop_pct": float(stop_pct),
        }
        st.caption("收盤跌破停損或跌回均線下才出場。判斷在收盤，成交在下一根開盤，不是盤中觸價。")
    elif strategy_name == "橫截面多因子 (Cross-Section)":
        xs_top_n = st.slider("持有前 N 名", 1, max(len(universe_list), 1), min(3, len(universe_list)))
        xs_weight = st.radio("權重", ["等權", "波動倒數"], horizontal=True)
        xs_months = st.slider("每隔幾個月的月末再平衡", 1, 6, 1)
        cross_section_params = {
            "top_n": int(xs_top_n),
            "weighting": "equal" if xs_weight == "等權" else "risk",
            "rebalance_months": int(xs_months),
            "momentum_lookback": 252,
            "momentum_skip": 21,
            "vol_lookback": 63,
            "momentum_weight": 0.5,
            "value_weight": 0.5,
        }
        st.caption(
            "每個再平衡日只用當天已公開的財報和當天以前的收盤。"
            "缺 PE 或動能的標的這期不排。權重在收盤決定，下一根開盤才調。"
            "波動倒數不是優化器。"
        )
    elif strategy_name == "核心衛星 (Core-Satellite)":
        core_weight = st.slider("核心權重", 0.50, 0.90, 0.70, step=0.05)
        core_choice = st.selectbox("核心", ["買進持有", "慢速趨勢停損"])
        satellite_choice = st.selectbox(
            "衛星",
            ["趨勢停損", "均線交叉", "RSI", "布林通道", "MACD", "橫截面多因子"],
        )
        core_symbol = st.text_input("核心代碼（空白＝跟衛星同一池）", value="SPY")
        core_key = "buy_and_hold" if core_choice == "買進持有" else "trend_stop"
        satellite_key = {
            "趨勢停損": "trend_stop",
            "均線交叉": "ma_cross",
            "RSI": "rsi",
            "布林通道": "bollinger",
            "MACD": "macd",
            "橫截面多因子": "cross_section",
        }[satellite_choice]
        if satellite_key == "trend_stop":
            satellite_params = {"trend_window": 80, "stop_mode": "percent", "stop_pct": 0.08}
        elif satellite_key == "cross_section":
            satellite_params = {
                "top_n": min(3, len(universe_list)),
                "weighting": "equal",
                "rebalance_months": 1,
                "momentum_lookback": 252,
                "momentum_skip": 21,
                "momentum_weight": 0.5,
                "value_weight": 0.5,
            }
        elif satellite_key == "ma_cross":
            satellite_params = {"short_window": 20, "long_window": 60, "ma_type": "SMA"}
        elif satellite_key == "rsi":
            satellite_params = {"period": 14, "oversold": 30, "overbought": 70}
        elif satellite_key == "bollinger":
            satellite_params = {"period": 20, "std_dev": 2.0}
        else:
            satellite_params = {"fast": 12, "slow": 26, "signal_period": 9}
        core_satellite_params = {
            "core_weight": float(core_weight),
            "satellite_weight": float(round(1.0 - core_weight, 10)),
            "core": core_key,
            "satellite": satellite_key,
            "core_symbol": core_symbol.strip().upper() or None,
            "core_params": (
                {}
                if core_key == "buy_and_hold"
                else {"trend_window": 150, "stop_mode": "percent", "stop_pct": 0.12}
            ),
            "satellite_params": satellite_params,
        }
        st.caption("70/30 這類比例只在一開始切資金。之後兩袖各自漲跌，不拉回，也沒有優化器。")

    st.markdown("### 樣本外驗證")
    use_walk_forward = st.checkbox(
        "Walk-forward 驗證",
        value=False,
        help="關掉時維持原本的整段回測。打開後只報告滾動測試窗，不把整段樣本內的淨值當結果。",
    )
    if use_walk_forward:
        train_months = int(
            st.number_input("訓練窗（月）", min_value=1, max_value=120, value=24, step=1)
        )
        test_months = int(
            st.number_input("測試窗（月）", min_value=1, max_value=36, value=6, step=1)
        )
        gap_days = int(
            st.number_input(
                "禁運間隔（日曆日）",
                min_value=1,
                max_value=120,
                value=21,
                step=1,
                help="訓練結束到測試開始，中間這幾天不拿來選參，也不計入該折的測試損益。",
            )
        )
        holdout_months = int(
            st.number_input(
                "末端鎖定 holdout（月，0 表示不留）",
                min_value=0,
                max_value=60,
                value=12,
                step=1,
                help="從結束日往前留一段。這段不進任何折，預設也不會被計算。",
            )
        )
        open_holdout = bool(
            st.checkbox(
                "打開鎖定的 holdout",
                value=False,
                disabled=holdout_months == 0,
                help="只另表報告最後一折已選定的參數。不要用這段結果回頭改格點或視窗。",
            )
        )
        if holdout_months == 0:
            open_holdout = False
        st.caption(
            "有參數的策略在內建小格點上選，買進持有不搜尋。上面的單一參數只用於一般回測。"
            "選股日是每一折訓練窗的第一天，用當時已公開的財報，不用今天的快照。"
            "同一折的訓練、禁運、測試不重疊；測試窗首尾相接。"
            "格點多於一組時，樣本外淨值會附上 Deflated Sharpe，訓練窗的 p 值做 Holm。"
        )
    else:
        train_months = 24
        test_months = 6
        gap_days = 21
        holdout_months = 0
        open_holdout = False

    st.markdown("---")
    run_btn = st.button("開始回測", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# 主畫面 Hero
# ──────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>量化金融回測系統</h1>
  <p>Leo project</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 未執行時顯示說明
# ──────────────────────────────────────────────
if not run_btn:
    st.info("請在左側設定回測參數，然後點擊「開始回測」")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "**🇺🇸 美股 / ETF**\n"
            "- `SPY` — S&P 500 ETF\n"
            "- `QQQ` — Nasdaq ETF\n"
            "- `AAPL` `MSFT` `GOOGL`\n"
            "- `GLD` `TLT` `VTI`"
        )
    with c2:
        st.markdown(
            "**指數**\n"
            "- `^GSPC` — S&P 500\n"
            "- `^IXIC` — Nasdaq\n"
            "- `^DJI` — 道瓊\n"
            "- `^N225` — 日經"
        )
    with c3:
        st.markdown(
            "**台股**\n"
            "- `0050.TW` — 台灣 50\n"
            "- `0056.TW` — 高股息\n"
            "- `2330.TW` — 台積電\n"
            "- `2317.TW` — 鴻海"
        )
    with c4:
        st.markdown(
            "**加密貨幣**\n"
            "- `BTC-USD` — 比特幣\n"
            "- `ETH-USD` — 以太幣\n"
            "- `SOL-USD` — Solana\n"
            "- `BNB-USD` — BNB"
        )
    st.stop()

if use_walk_forward:
    render_walk_forward(
        universe=universe_list,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        risk_free_rate=risk_free_rate,
        strategy_name=strategy_name,
        train_months=train_months,
        test_months=test_months,
        gap_days=gap_days,
        holdout_months=holdout_months,
        open_holdout=open_holdout,
        use_live_snapshot=use_live_snapshot,
        use_screener=use_screener,
        pe_max=pe_max,
        roe_min=roe_min,
        rev_growth_min=rev_growth_min,
        mcap_min_b=mcap_min_b,
        use_factor_scoring=use_factor_scoring,
        active_factors=active_factors,
        factor_top_n=factor_top_n,
        norm_method_key=norm_method_key,
    )
    st.stop()

# ──────────────────────────────────────────────
# 執行與回測邏輯
# ──────────────────────────────────────────────
filtered_symbols = universe_list
strategy_key = canonical_strategy(strategy_name)
satellite_key = str(core_satellite_params.get("satellite") or "")
owns_cross_section = strategy_key == "cross_section" or (
    strategy_key == "core_satellite" and satellite_key == "cross_section"
)
needs_fundamentals = (not owns_cross_section) and (
    use_screener or (use_factor_scoring and bool(active_factors))
)
if owns_cross_section and (use_screener or use_factor_scoring or use_live_snapshot):
    st.caption(
        "橫截面排名用每次再平衡日已公開的財報，側邊欄的一次性篩選和今天的快照這次不套用。"
    )

# Step 1: Hard-threshold fundamental filter.
# Historical runs use statements knowable on the start date. The live
# snapshot is opt-in and labeled as today's numbers.
if needs_fundamentals:
    if use_live_snapshot:
        st.warning(
            "基本面用的是 Yahoo 今天的快照，不是回測起始日看得到的數字。"
            "這樣選出來的股票含有前視，不能當成歷史結果。"
        )
        with st.spinner("正在取得 Yahoo 今日基本面快照..."):
            df_fun = fetch_fundamentals(universe_list)
    else:
        lag_days = reporting_lag_days()
        st.caption(
            f"基本面對齊 {start_date:%Y-%m-%d}：只用這天以前已公開的財報。"
            f"沒有法說日或申報日時，假設會計期間結束後 {lag_days} 天才看得到。"
            "動能與波動率也只算到這一天的收盤。"
        )
        with st.spinner("正在依回測起始日整理當時可知的基本面..."):
            df_fun = fetch_fundamentals_pit(universe_list, start_date, lag_days=lag_days)

    if use_screener:
        mcap_real = mcap_min_b * 1_000_000_000 if mcap_min_b else None
        pe_val = pe_max if pe_max > 0 else None

        filtered_df = filter_stocks(
            df_fun,
            pe_max=pe_val,
            roe_min=roe_min if roe_min > -10 else None,
            rev_growth_min=rev_growth_min if rev_growth_min > -10 else None,
            mcap_min=mcap_real,
        )
        filtered_symbols = filtered_df["symbol"].tolist()

        st.markdown("### 📊 基本面篩選結果")
        st.dataframe(filtered_df, use_container_width=True)
else:
    df_fun = None

if not filtered_symbols:
    if use_screener and not use_live_snapshot:
        st.error(
            "沒有股票通過基本面篩選。較早的日期若還沒有已公開的財報，相關欄位會是空值並被剔除。"
            "可以放寬條件，或改勾今天的快照（那不是歷史結果）。"
        )
    else:
        st.error("沒有股票通過基本面篩選，請放寬條件。")
    st.stop()

# Factor scoring re-ranks the filtered universe and keeps the top N.
# A cross-section book ranks itself on each rebalance, so this one-shot
# score is not applied on top of it.
if (not owns_cross_section) and use_factor_scoring and active_factors:
    with st.spinner("正在計算因子分數並排名..."):
        # Only score stocks that survived the hard filter
        df_fun_filtered = df_fun[df_fun["symbol"].isin(filtered_symbols)]
        ranked_df = apply_factor_score(
            df_fundamentals=df_fun_filtered,
            as_of=start_date,
            active_factors=active_factors,
            top_n=int(factor_top_n),
            method=norm_method_key,
        )
        filtered_symbols = ranked_df["symbol"].tolist()

        # ── Visualisation: Factor Distribution ─────────────────
        st.markdown("### 🏆 因子評分排名")
        # Score bar chart (go is already imported at the top of the file)
        fig_score = go.Figure(
            go.Bar(
                x=ranked_df["symbol"],
                y=ranked_df["Composite_Score"],
                marker_color=[
                    COLOR_BUY if v >= 0 else COLOR_SELL
                    for v in ranked_df["Composite_Score"]
                ],
                text=[f"{v:.2f}" for v in ranked_df["Composite_Score"]],
                textposition="outside",
            )
        )
        fig_score.update_layout(
            title="Top N 股票綜合因子分數",
            xaxis_title="股票代號",
            yaxis_title="Composite Score",
            template=PLOTLY_THEME,
            height=320,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # Detail table — show factor z-scores
        display_cols = ["symbol", "Composite_Score"] + [
            c for c in ranked_df.columns if c.startswith("z_")
        ]
        st.dataframe(
            ranked_df[[c for c in display_cols if c in ranked_df.columns]],
            use_container_width=True,
            hide_index=True,
        )

run_config = RunConfig(
    strategy=strategy_name,
    tickers=list(filtered_symbols),
    start=str(pd.Timestamp(start_date).date()),
    end=str(pd.Timestamp(end_date).date()),
    benchmark=_benchmark_symbol(benchmark, filtered_symbols),
    initial_capital=float(initial_capital),
    commission=float(commission),
    slippage=float(slippage),
    risk_free_rate=float(risk_free_rate),
    walk_forward=False,
    params=dict(strategy_params),
    cross_section=dict(cross_section_params),
    core_satellite=dict(core_satellite_params),
)
try:
    with st.spinner("下載歷史報價並執行策略回測…"):
        research = run_research(run_config)
except ResearchError as exc:
    st.error(str(exc))
    st.stop()
except WalkForwardError as exc:
    st.error(str(exc))
    st.stop()

for note in research.notes:
    st.caption(note)

equity_curve = research.equity
returns = research.returns
trades = research.trades
metrics_disp = research.metrics_display
metrics_raw = research.metrics
monthly_ret = research.monthly_returns
annual_ret = research.annual_returns
data = research.chart_prices if research.chart_prices is not None else pd.DataFrame()

if research.holdings is not None and not research.holdings.empty:
    st.markdown("#### 再平衡當日可知的排名")
    st.caption(
        "as_of 是決定權重的收盤日。動能只用該日以前的收盤；PE 來自 point-in-time，不是今天的 Yahoo 快照。"
        "成交在下一根開盤。"
    )
    st.dataframe(research.holdings, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 標題列
# ──────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
with h1:
    if strategy_key == "cross_section":
        book_title = f"橫截面　｜　{strategy_name}"
    elif strategy_key == "core_satellite":
        book_title = f"核心衛星　｜　{strategy_name}"
    else:
        book_title = f"等權重投資組合 ({len(filtered_symbols)} 檔)　｜　{strategy_name}"
    st.markdown(f"### {book_title}")
with h2:
    csv_bytes = (
        pd.DataFrame(list(metrics_disp.items()), columns=["指標", "數值"])
        .to_csv(index=False)
        .encode("utf-8-sig")
    )
    st.download_button("下載報告", csv_bytes, f"portfolio_report.csv", "text/csv")

st.markdown("---")


# ──────────────────────────────────────────────
# KPI 卡片
# ──────────────────────────────────────────────
tr = metrics_raw["total_return"]
sh = metrics_raw["sharpe"]
md = metrics_raw["max_drawdown"]
cg = metrics_raw["cagr"]
wr = metrics_raw["win_rate"]
vl = metrics_raw["annual_vol"]

cards_html = (
    '<div class="kpi-grid">'
    + kpi_card("總報酬率", f"{tr:.2%}", f"CAGR {cg:.2%}", sign_class(tr))
    + kpi_card("夏普比率", f"{sh:.2f}", ">1 佳  >2 優秀", sign_class(sh))
    + kpi_card("最大回撤", f"{md:.2%}", f"Calmar {metrics_raw['calmar']:.2f}", sign_class(md, False))
    + kpi_card("年化報酬", f"{cg:.2%}", "", sign_class(cg))
    + kpi_card("勝率", f"{wr:.1%}", f"獲利/虧損 {metrics_raw['win_loss_ratio']:.2f}x", sign_class(wr - 0.5))
    + kpi_card("年化波動", f"{vl:.2%}", f"VaR95% {metrics_raw['var_95']:.2%}", "neg" if vl > 0.25 else "neu")
    + "</div>"
)
st.markdown(cards_html, unsafe_allow_html=True)
_render_rigor(research, expanded=bool(research.selection and research.selection.get("multiple_testing")))

# ──────────────────────────────────────────────
# 頁籤
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["績效總覽", "詳細分析", "風險分析", "交易記錄"]
)

# ══════════════════════════════════════════════
# TAB 1 — 績效總覽
# ══════════════════════════════════════════════
with tab1:
    st.plotly_chart(research.equity_figure, use_container_width=True)
    st.plotly_chart(research.drawdown_figure, use_container_width=True)

    # ── 年度報酬條形圖 ──
    if not annual_ret.empty:
        colors = [COLOR_BUY if v >= 0 else COLOR_SELL for v in annual_ret.values]
        fig_ann = go.Figure(
            go.Bar(
                x=annual_ret.index.year.astype(str),
                y=annual_ret.values * 100,
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in annual_ret.values],
                textposition="outside",
            )
        )
        fig_ann.update_layout(
            title="年度報酬率",
            xaxis_title="年份",
            yaxis_title="報酬率 (%)",
            template=PLOTLY_THEME,
            height=300,
            margin=dict(t=50, b=40),
        )
        fig_ann.add_hline(y=0, line_color="#8B949E", line_width=1)
        st.plotly_chart(fig_ann, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — 詳細分析
# ══════════════════════════════════════════════
with tab2:
    left, right = st.columns([1, 2])

    with left:
        st.markdown("#### 完整績效指標")
        df_metrics = pd.DataFrame(
            list(metrics_disp.items()), columns=["指標", "數值"]
        )
        st.dataframe(df_metrics, use_container_width=True, hide_index=True, height=580)

    with right:
        # 月度熱力圖
        if not monthly_ret.empty:
            month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                            "7月", "8月", "9月", "10月", "11月", "12月"]
            z_vals = monthly_ret.values * 100
            text_vals = np.where(
                np.isnan(z_vals), "", np.round(z_vals, 1).astype(str) + "%"
            )
            fig_heat = go.Figure(
                go.Heatmap(
                    z=z_vals,
                    x=month_labels,
                    y=monthly_ret.index.astype(str),
                    colorscale=[[0, "#F85149"], [0.5, "#21262D"], [1, "#3FB950"]],
                    zmid=0,
                    text=text_vals,
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(title="報酬率 (%)"),
                )
            )
            fig_heat.update_layout(
                title="月度報酬率熱力圖",
                template=PLOTLY_THEME,
                height=340,
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # 報酬分布
        st.markdown("#### 日報酬率分布")
        r_clean = returns.replace([np.inf, -np.inf], np.nan).dropna() * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=r_clean,
                nbinsx=60,
                name="日報酬率",
                marker_color=COLOR_STRATEGY,
                opacity=0.7,
            )
        )
        var95_pct = metrics_raw["var_95"] * 100
        fig_dist.add_vline(
            x=var95_pct,
            line_dash="dash",
            line_color=COLOR_SELL,
            annotation_text=f"VaR 95%: {var95_pct:.2f}%",
            annotation_position="top left",
        )
        fig_dist.update_layout(
            xaxis_title="日報酬率 (%)",
            yaxis_title="頻率",
            template=PLOTLY_THEME,
            height=280,
            margin=dict(t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 滾動夏普比率 (252 天)")
    st.plotly_chart(research.rolling_sharpe_figure, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — 風險分析
# ══════════════════════════════════════════════
with tab3:
    r3a, r3b = st.columns(2)

    with r3a:
        # VaR 圖
        r_pct = returns.replace([np.inf, -np.inf], np.nan).dropna() * 100
        var95 = metrics_raw["var_95"] * 100
        cvar95 = metrics_raw["cvar_95"] * 100
        fig_var = go.Figure()
        fig_var.add_trace(
            go.Histogram(
                x=r_pct, nbinsx=60,
                marker_color=COLOR_STRATEGY, opacity=0.6, name="報酬分布",
            )
        )
        fig_var.add_vline(x=var95, line_dash="dash", line_color=COLOR_SELL,
                          annotation_text=f"VaR 95%: {var95:.2f}%")
        fig_var.add_vline(x=cvar95, line_dash="dash", line_color="#FF0000",
                          annotation_text=f"CVaR 95%: {cvar95:.2f}%",
                          annotation_position="bottom right")
        fig_var.update_layout(
            title="風險值 (VaR / CVaR) 分析",
            xaxis_title="日報酬率 (%)", yaxis_title="頻率",
            template=PLOTLY_THEME, height=360, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with r3b:
        # 滾動波動率
        roll_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=roll_vol.index, y=roll_vol,
                name="30 天滾動波動率",
                line=dict(color=COLOR_BENCHMARK, width=2),
                fill="tozeroy", fillcolor="rgba(240,136,62,0.12)",
            )
        )
        fig_vol.update_layout(
            title="滾動年化波動率 (30 天)",
            xaxis_title="日期", yaxis_title="年化波動率 (%)",
            template=PLOTLY_THEME, height=360, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # 水下曲線（回撤持續期間）
    fig_uw = go.Figure()
    fig_uw.add_trace(
        go.Scatter(
            x=dd_pct.index, y=dd_pct,
            name="水下期間",
            line=dict(color=COLOR_SELL, width=1),
            fill="tozeroy", fillcolor=COLOR_DD,
        )
    )
    fig_uw.update_layout(
        title="水下曲線（資金回撤持續期間）",
        xaxis_title="日期", yaxis_title="回撤 (%)",
        template=PLOTLY_THEME, height=260, margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_uw, use_container_width=True)

    # 風險摘要
    st.markdown("#### 風險指標摘要")
    risk_table = pd.DataFrame(
        {
            "指標": ["年化波動率", "VaR (95%)", "CVaR (95%)", "最大回撤", "偏度", "超額峰度"],
            "數值": [
                f"{metrics_raw['annual_vol']:.2%}",
                f"{metrics_raw['var_95']:.2%}",
                f"{metrics_raw['cvar_95']:.2%}",
                f"{metrics_raw['max_drawdown']:.2%}",
                f"{returns.skew():.3f}",
                f"{returns.kurt():.3f}",
            ],
            "說明": [
                "年化標準差，衡量價格波動程度",
                "95% 信心水準下，單日最大可能損失",
                "超過 VaR 的平均損失（尾部風險）",
                "從高點至低點的最大跌幅",
                "報酬分布偏斜（負值表示左偏，下跌尾較重）",
                "峰態（正值表示厚尾，極端事件更多）",
            ],
        }
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — 交易記錄
# ══════════════════════════════════════════════
with tab4:
    if trades.empty:
        st.info("此策略沒有具體的交易記錄（如買進持有策略只有一次買入）。")
    else:
        n_trades = len(trades)
        n_rounds = n_trades // 2
        st.markdown(f"#### 交易記錄　｜　共 **{n_trades}** 筆，**{n_rounds}** 個完整回合")

        # 計算每回合損益
        if n_rounds > 0:
            buy_list = trades[trades["action"] == "BUY"].reset_index(drop=True)
            sell_list = trades[trades["action"] == "SELL"].reset_index(drop=True)
            pairs = min(len(buy_list), len(sell_list))
            if pairs > 0:
                pnl = sell_list["value"].iloc[:pairs].values - buy_list["value"].iloc[:pairs].values
                wins = (pnl > 0).sum()
                losses = (pnl <= 0).sum()
                c4a, c4b, c4c, c4d = st.columns(4)
                c4a.metric("總交易次數", n_rounds)
                c4b.metric("獲利次數", wins)
                c4c.metric("虧損次數", losses)
                c4d.metric("策略勝率", f"{wins/pairs:.1%}")

        # 交易表格
        td = trades.copy()
        td["date"] = pd.to_datetime(td["date"]).dt.strftime("%Y-%m-%d")
        td["price"] = td["price"].map("${:.2f}".format)
        td["shares"] = td["shares"].map("{:.4f}".format)
        td["value"] = td["value"].map("${:,.2f}".format)
        show_fee = "fee" in td.columns
        if show_fee:
            td["fee"] = td["fee"].map("${:,.2f}".format)

        columns = ["date"]
        headers = ["日期"]
        if "symbol" in td.columns:
            columns.append("symbol")
            headers.append("標的")
        columns.extend(["action", "price", "shares", "value"])
        headers.extend(["動作", "成交價", "股數", "金額"])
        if show_fee:
            columns.append("fee")
            headers.append("費用")
        td = td[columns]
        td.columns = headers

        def highlight_action(row):
            color = "#1A3A1A" if row["動作"] == "BUY" else "#3A1A1A"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            td.style.apply(highlight_action, axis=1),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        # 標記只畫在商品池第一檔的價格上，避免把別檔的成交點到這條線上。
        st.markdown("#### 交易信號圖")
        plot_trades = trades
        if research.chart_symbol and "symbol" in trades.columns:
            plot_trades = trades[trades["symbol"] == research.chart_symbol]
            st.caption(f"價格是 {research.chart_symbol}。其他標的的成交在上面的表格。")
        if data is None or getattr(data, "empty", True) or "Close" not in data.columns:
            st.caption("這一段沒有對得上的價格，所以不畫信號圖。")
        else:
            _draw_signal_chart(data, plot_trades)


# ──────────────────────────────────────────────
# 頁尾
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8B949E;font-size:0.8rem;'>"
    "資料來源：Yahoo Finance"
    "</p>",
    unsafe_allow_html=True,
)
