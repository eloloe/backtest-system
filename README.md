# 量化金融回測系統 (Quant Backtesting System)

一個以 **Streamlit** 打造的互動式量化策略回測平台，整合基本面篩選、多因子評分、技術策略回測與完整績效／風險分析，可一站式驗證選股與交易策略。

**線上 Demo**：https://backtest-system-gfba5qpzyz3hzc6jbsuj2u.streamlit.app/

---

## 功能特色

- **商品池設定**：支援美股、ETF、指數、台股（`.TW`）、加密貨幣（`BTC-USD` 等）。預設資料來源為 Yahoo Finance，可改接 Tiingo 或 Polygon（見下方）。
- **基本面濾網**：以 PE、ROE、營收成長率、市值等硬門檻篩選股票池。
- **多因子評分系統**：ROE / PE / PB / 營收成長 / 盈餘成長 / 動能 / 波動率等因子，可自由開關與調整權重，支援 Z-Score 與百分位排名兩種標準化，選出綜合分數 Top N。
- **五種交易策略**：
  - 買進持有 (Buy & Hold)
  - 均線交叉 (MA Cross，SMA / EMA)
  - RSI 策略
  - 布林通道 (Bollinger Bands)
  - MACD 策略
- **事件驅動回測引擎**：考慮手續費與滑價，以等權重投資組合方式回測多檔標的。
- **完整績效與風險分析**：
  - 報酬指標：總報酬、CAGR、年度／月度報酬熱力圖
  - 風險調整：Sharpe、Sortino、Calmar
  - 風險指標：最大回撤、水下曲線、年化波動率、VaR / CVaR、偏度、峰度
  - 相對指標：對基準指數的 Alpha / Beta
  - 滾動 Sharpe、滾動波動率、交易信號圖與交易明細

---

## 專案結構

| 檔案 | 說明 |
|------|------|
| `app.py` | Streamlit 主程式（UI、流程整合、圖表） |
| `data_fetcher.py` | 依設定的 provider 取得 OHLCV |
| `providers/` | 價格來源（Yahoo / Tiingo / Polygon）與本機 Parquet 快取 |
| `fundamentals.py` | 抓取基本面資料（PE、PB、ROE…，目前仍是 Yahoo 當下快照） |
| `screener.py` | 硬門檻篩選與因子評分整合 |
| `factor_engine.py` | 因子註冊表、價格因子計算、標準化與綜合評分 |
| `strategies.py` | 各項技術交易策略的訊號產生邏輯 |
| `backtester.py` | 事件驅動回測引擎（含手續費／滑價） |
| `metrics.py` | 績效與風險指標計算 |


---

## 資料來源

股價走 `providers/`。用環境變數 `BACKTEST_DATA_PROVIDER` 選擇，預設 `yahoo`，不需要金鑰就能跑 Demo。

| 名稱 | 金鑰 | 涵蓋範圍 | 價格調整 |
|------|------|----------|----------|
| `yahoo`（預設） | 無 | 沿用 Yahoo 代碼：美股、ETF、指數、台股（`.TW`）、加密貨幣（`BTC-USD`） | `auto_adjust=True`，除權息 |
| `tiingo` | `TIINGO_API_KEY` | 美股、ETF 日線 | 使用 `adj*` 欄位（除權息） |
| `polygon` | `POLYGON_API_KEY` | 美股、ETF 日線 | `adjusted=true`，只還原拆股，不含股息 |

切換範例：

```bash
export BACKTEST_DATA_PROVIDER=tiingo
export TIINGO_API_KEY=你的金鑰
streamlit run app.py
```

Polygon 把 `BACKTEST_DATA_PROVIDER` 設成 `polygon`，金鑰變數是 `POLYGON_API_KEY`。選了這兩個來源卻沒設金鑰時會直接報錯，不會悄悄退回 Yahoo，專案裡也不會放任何金鑰。

成功取回的日線會寫進 `data_cache/`（Parquet，已列入 `.gitignore`）。鍵是 provider、代碼、起始日與結束日。關掉快取：`BACKTEST_DATA_CACHE=0`。換目錄：`BACKTEST_DATA_CACHE_DIR`。

`end` 不含當日，跟 `yfinance` 的 `history(end=...)` 一樣。

目前的限制：

- Yahoo 適合示範。調整方式與服務穩定性都不該當成研究級資料。
- Tiingo 與 Polygon 這一步只接美股／ETF 日線。代碼原樣送出，不會把 `2330.TW` 或 `BTC-USD` 轉成其他市場的格式。台股與加密貨幣的來源以後再加。
- Polygon 沒有做股息調整，收盤價不會和 Yahoo／Tiingo 的總報酬價格對齊。
- 基本面仍是 Yahoo `ticker.info` 的當下快照，拿來做歷史篩選有前視問題。這一步沒有改成 point-in-time。

跑單元測試（不需要金鑰）：

```bash
pip install -r requirements-dev.txt
pytest
```

## 免責聲明

本專案僅供教育與研究用途，所有回測結果基於歷史資料，**不構成任何投資建議**。歷史績效不代表未來表現，實際交易存在滑價、流動性、稅務等成本，請自行評估風險。

---

_預設資料來源：Yahoo Finance_
