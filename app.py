
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester import Backtester
from data_fetcher import fetch_data
from metrics import calculate_annual_returns, calculate_metrics, calculate_monthly_returns
from strategies import (
    BollingerBandsStrategy,
    BuyAndHold,
    MACDStrategy,
    MACross,
    RSIStrategy,
)

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
.stApp { background-color: #F97316; }

/* 側邊欄 */
section[data-testid="stSidebar"] { background-color: #161B22; }

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
]

PLOTLY_THEME = "plotly_dark"
COLOR_STRATEGY = "#58A6FF"
COLOR_BENCHMARK = "#F0883E"
COLOR_BUY = "#3FB950"
COLOR_SELL = "#F85149"
COLOR_DD = "rgba(248, 81, 73, 0.25)"

# ──────────────────────────────────────────────
# 側邊欄
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 回測設定")
    st.markdown("---")

    st.markdown("### 📊 商品設定")
    symbol = st.text_input(
        "交易標的 (Yahoo Finance 代碼)",
        value="SPY",
        help="例如：SPY、AAPL、^GSPC、BTC-USD、0050.TW",
    )
    benchmark = st.text_input(
        "基準指數（選填）",
        value="SPY",
        help="用於計算 Alpha / Beta，留空則跳過",
    )

    st.markdown("### 📅 回測期間")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("開始日期", value=datetime(2018, 1, 1))
    with col_b:
        end_date = st.date_input("結束日期", value=datetime.today())

    st.markdown("### 💰 資金設定")
    initial_capital = st.number_input(
        "初始資金 (USD)", value=100_000, min_value=1_000, step=10_000
    )
    commission = (
        st.slider("手續費 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    )
    slippage = (
        st.slider("滑價 (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100
    )
    risk_free_rate = (
        st.slider("無風險利率 (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100
    )

    st.markdown("### 🎯 交易策略")
    strategy_name = st.selectbox("選擇策略", STRATEGY_LIST)

    # 策略參數
    if strategy_name == "均線交叉 (MA Cross)":
        ma_type = st.radio("均線類型", ["SMA", "EMA"], horizontal=True)
        short_window = st.slider("短期均線", 5, 100, 20)
        long_window = st.slider("長期均線", 20, 300, 60)
    elif strategy_name == "RSI 策略":
        rsi_period = st.slider("RSI 週期", 2, 50, 14)
        rsi_oversold = st.slider("超賣門檻", 10, 45, 30)
        rsi_overbought = st.slider("超買門檻", 55, 90, 70)
    elif strategy_name == "布林通道 (Bollinger Bands)":
        bb_period = st.slider("週期", 5, 50, 20)
        bb_std = st.slider("標準差倍數", 1.0, 3.5, 2.0, step=0.1)
    elif strategy_name == "MACD 策略":
        macd_fast = st.slider("快線週期", 5, 30, 12)
        macd_slow = st.slider("慢線週期", 15, 60, 26)
        macd_signal_p = st.slider("信號線週期", 3, 20, 9)

    st.markdown("---")
    run_btn = st.button("🚀 開始回測", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# 主畫面 Hero
# ──────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>📈 量化金融回測系統</h1>
  <p>支援股票、ETF、指數、加密貨幣 | 5 種策略 | 17+ 績效指標 | 互動式圖表</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 未執行時顯示說明
# ──────────────────────────────────────────────
if not run_btn:
    st.info("👈 請在左側設定回測參數，然後點擊「🚀 開始回測」")
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
            "**📊 指數**\n"
            "- `^GSPC` — S&P 500\n"
            "- `^IXIC` — Nasdaq\n"
            "- `^DJI` — 道瓊\n"
            "- `^N225` — 日經"
        )
    with c3:
        st.markdown(
            "**🇹🇼 台股**\n"
            "- `0050.TW` — 台灣 50\n"
            "- `0056.TW` — 高股息\n"
            "- `2330.TW` — 台積電\n"
            "- `2317.TW` — 鴻海"
        )
    with c4:
        st.markdown(
            "**₿ 加密貨幣**\n"
            "- `BTC-USD` — 比特幣\n"
            "- `ETH-USD` — 以太幣\n"
            "- `SOL-USD` — Solana\n"
            "- `BNB-USD` — BNB"
        )
    st.stop()

# ──────────────────────────────────────────────
# 資料下載
# ──────────────────────────────────────────────
with st.spinner(f"⏳ 正在下載 {symbol} 資料…"):
    data = fetch_data(symbol, str(start_date), str(end_date))

if data.empty:
    st.error(f"❌ 無法取得「{symbol}」的資料，請確認代碼是否正確。")
    st.stop()

# 基準指數
bench_equity = None
benchmark_returns = None
if benchmark and benchmark.strip():
    with st.spinner(f"⏳ 下載基準指數 {benchmark}…"):
        bench_data = fetch_data(benchmark.strip(), str(start_date), str(end_date))
    if not bench_data.empty:
        bench_r = bench_data["Close"].pct_change().dropna()
        benchmark_returns = bench_r
        bench_equity = (1 + bench_r).cumprod() * initial_capital

# ──────────────────────────────────────────────
# 建立策略
# ──────────────────────────────────────────────
with st.spinner("⏳ 執行策略回測…"):
    if strategy_name == "買進持有 (Buy & Hold)":
        strategy = BuyAndHold()
    elif strategy_name == "均線交叉 (MA Cross)":
        strategy = MACross(short_window=short_window, long_window=long_window, ma_type=ma_type)
    elif strategy_name == "RSI 策略":
        strategy = RSIStrategy(period=rsi_period, oversold=rsi_oversold, overbought=rsi_overbought)
    elif strategy_name == "布林通道 (Bollinger Bands)":
        strategy = BollingerBandsStrategy(period=bb_period, std_dev=bb_std)
    elif strategy_name == "MACD 策略":
        strategy = MACDStrategy(fast=macd_fast, slow=macd_slow, signal_period=macd_signal_p)

    signals = strategy.generate_signals(data)

    bt = Backtester(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )
    portfolio, trades = bt.run(data, signals)

    equity_curve = portfolio["portfolio_value"]
    returns = portfolio["returns"].dropna()

    metrics_disp, metrics_raw = calculate_metrics(
        equity_curve=equity_curve,
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    )
    monthly_ret = calculate_monthly_returns(equity_curve)
    annual_ret = calculate_annual_returns(equity_curve)

# ──────────────────────────────────────────────
# 標題列
# ──────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(f"### {symbol}　｜　{strategy_name}")
with h2:
    csv_bytes = (
        pd.DataFrame(list(metrics_disp.items()), columns=["指標", "數值"])
        .to_csv(index=False)
        .encode("utf-8-sig")
    )
    st.download_button("📥 下載報告", csv_bytes, f"{symbol}_report.csv", "text/csv")

st.markdown("---")


# ──────────────────────────────────────────────
# KPI 卡片
# ──────────────────────────────────────────────
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


tr = metrics_raw["total_return"]
sh = metrics_raw["sharpe"]
md = metrics_raw["max_drawdown"]
cg = metrics_raw["cagr"]
wr = metrics_raw["win_rate"]
vl = metrics_raw["annual_vol"]

cards_html = (
    '<div class="kpi-grid">'
    + kpi_card("📈 總報酬率", f"{tr:.2%}", f"CAGR {cg:.2%}", sign_class(tr))
    + kpi_card("⚡ 夏普比率", f"{sh:.2f}", ">1 佳  >2 優秀", sign_class(sh))
    + kpi_card("📉 最大回撤", f"{md:.2%}", f"Calmar {metrics_raw['calmar']:.2f}", sign_class(md, False))
    + kpi_card("🎯 年化報酬", f"{cg:.2%}", "", sign_class(cg))
    + kpi_card("✅ 勝率", f"{wr:.1%}", f"獲利/虧損 {metrics_raw['win_loss_ratio']:.2f}x", sign_class(wr - 0.5))
    + kpi_card("📊 年化波動", f"{vl:.2%}", f"VaR95% {metrics_raw['var_95']:.2%}", "neg" if vl > 0.25 else "neu")
    + "</div>"
)
st.markdown(cards_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 頁籤
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 績效總覽", "📊 詳細分析", "⚠️ 風險分析", "🔄 交易記錄"]
)

# ══════════════════════════════════════════════
# TAB 1 — 績效總覽
# ══════════════════════════════════════════════
with tab1:
    # ── 淨值曲線 ──
    fig_eq = go.Figure()

    # 策略淨值
    fig_eq.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve,
            name=f"{symbol} ({strategy_name})",
            line=dict(color=COLOR_STRATEGY, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.06)",
        )
    )

    # 基準淨值
    if bench_equity is not None:
        be = bench_equity.reindex(equity_curve.index).ffill()
        fig_eq.add_trace(
            go.Scatter(
                x=be.index,
                y=be,
                name=f"{benchmark} (基準)",
                line=dict(color=COLOR_BENCHMARK, width=2, dash="dash"),
            )
        )

    # 買賣標記
    if not trades.empty:
        buys = trades[trades["action"] == "BUY"]
        sells = trades[trades["action"] == "SELL"]
        if not buys.empty:
            fig_eq.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=equity_curve.reindex(buys["date"]).values,
                    mode="markers",
                    name="買入",
                    marker=dict(color=COLOR_BUY, size=10, symbol="triangle-up"),
                )
            )
        if not sells.empty:
            fig_eq.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=equity_curve.reindex(sells["date"]).values,
                    mode="markers",
                    name="賣出",
                    marker=dict(color=COLOR_SELL, size=10, symbol="triangle-down"),
                )
            )

    fig_eq.update_layout(
        title=f"{symbol} 投資組合淨值曲線",
        xaxis_title="日期",
        yaxis_title="投資組合價值 (USD)",
        template=PLOTLY_THEME,
        height=420,
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── 回撤曲線 ──
    dd_pct = (equity_curve - equity_curve.cummax()) / equity_curve.cummax() * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd_pct.index,
            y=dd_pct,
            name="回撤",
            line=dict(color=COLOR_SELL, width=1.5),
            fill="tozeroy",
            fillcolor=COLOR_DD,
        )
    )
    fig_dd.update_layout(
        title="回撤分析 (Drawdown)",
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        template=PLOTLY_THEME,
        height=240,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

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
        st.markdown("#### 📋 完整績效指標")
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

    # 滾動夏普
    st.markdown("#### 滾動夏普比率 (252 天)")
    roll_r = returns.replace([np.inf, -np.inf], np.nan)
    rolling_sharpe = roll_r.rolling(252).apply(
        lambda x: (x.mean() * 252 - risk_free_rate) / (x.std() * np.sqrt(252))
        if x.std() > 0
        else 0
    )
    fig_rs = go.Figure()
    fig_rs.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            name="滾動夏普",
            line=dict(color="#BC8CFF", width=2),
        )
    )
    fig_rs.add_hline(y=1, line_dash="dot", line_color=COLOR_BUY, annotation_text="良好 (>1)")
    fig_rs.add_hline(y=0, line_dash="dot", line_color="#8B949E")
    fig_rs.update_layout(
        yaxis_title="夏普比率",
        template=PLOTLY_THEME,
        height=280,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_rs, use_container_width=True)

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
        td.columns = ["日期", "動作", "成交價", "股數", "金額"]

        def highlight_action(row):
            color = "#1A3A1A" if row["動作"] == "BUY" else "#3A1A1A"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            td.style.apply(highlight_action, axis=1),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        # 交易圖表
        st.markdown("#### 交易信號圖")
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
        buys = trades[trades["action"] == "BUY"]
        sells = trades[trades["action"] == "SELL"]
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

# ──────────────────────────────────────────────
# 頁尾
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8B949E;font-size:0.8rem;'>"
    "資料來源：Yahoo Finance　｜　本系統僅供學術研究與回測分析，不構成任何投資建議"
    "</p>",
    unsafe_allow_html=True,
)
