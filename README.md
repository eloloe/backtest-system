# 量化金融回測系統 (Quant Backtesting System)

一個以 **Streamlit** 打造的互動式量化策略回測平台，整合基本面篩選、多因子評分、技術策略回測與完整績效／風險分析，可一站式驗證選股與交易策略。

**線上 Demo**：https://backtest-system-gfba5qpzyz3hzc6jbsuj2u.streamlit.app/

---

## 功能特色

- **商品池設定**：支援美股、ETF、指數、台股（`.TW`）、加密貨幣（`BTC-USD` 等）。預設資料來源為 Yahoo Finance，可改接 Tiingo 或 Polygon（見下方）。
- **基本面濾網**：歷史回測用決策日（回測起始日）的時點 PE、ROE、營收成長、市值。互動畫面可改看 Yahoo 當下快照，並會標明那不是歷史資料。
- **多因子評分系統**：ROE / PE / PB / 營收成長 / 盈餘成長 / 動能 / 波動率等因子，可自由開關與調整權重，支援 Z-Score 與百分位排名兩種標準化，選出綜合分數 Top N。時點模式下，動能與波動只算到決策日為止。
- **五種交易策略**：
  - 買進持有 (Buy & Hold)
  - 均線交叉 (MA Cross，SMA / EMA)
  - RSI 策略
  - 布林通道 (Bollinger Bands)
  - MACD 策略
- **事件驅動回測引擎**：訊號在收盤決定、下一根開盤成交。美股用手續費與滑價，台股用券商手續費與賣出證交稅。多檔標的仍是等權重、各檔全進全出。
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
| `fundamentals/` | 當下快照，以及用財報與決策日價格算出的時點基本面 |
| `screener.py` | 硬門檻篩選與因子評分整合 |
| `factor_engine.py` | 因子註冊表、價格因子計算、標準化與綜合評分 |
| `strategies.py` | 各項技術交易策略的訊號產生邏輯 |
| `backtester.py` | 回測引擎（下一根開盤成交、美股／台股成本） |
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
## 基本面時點

歷史回測的選股預設走時點。決策日是回測起始日，而且目前只在起始日選一次，沒有定期再平衡。PE、PB、ROE、營收成長、盈餘成長、市值只用該日以前看得到的資料。

預設來源是 Yahoo 的季度與年度財報，加上決策日當天、或該日之前最後一個交易日的收盤價（往前最多看 30 個日曆天），由程式自己算比率：

- 有連續四季時用近四季（TTM）。中間缺一季就改用最近的年報，不會把不連續的四季加總。
- 成長率要有前一段 TTM，或上一財年。前一期的基數要是正的。
- PE 只用正的稀釋 EPS。PB 與 ROE 只用正的股東權益。
- 市值是決策日收盤價乘上最近的流通股數；沒有流通股數時用稀釋加權平均股數。

Yahoo 通常沒有申報日，所以假設財報在會計期間結束後 **60 個日曆天**才公開。這個天數是假設，不是交易所規則。用 `BACKTEST_FUNDAMENTALS_LAG_DAYS` 改。若 Yahoo 的法說日對得上該季（期間結束後 120 天內，而且已經有公布 EPS），就用法說日，不再加這 60 天。FMP 有 `acceptedDate` 或 `filingDate` 時用那個日期，同樣不再加固定天數。申報日只取日曆日，不看幾點鐘；盤後公布仍可能被算進當天。

這不是 Compustat 或 TEJ。Yahoo 每一期只給「現在看得到的那一版」。公司若已重編、而且 Yahoo 把舊數字蓋掉了，較早的決策日仍可能吃到重編後的數字。選股邏輯會擋掉 `available_at` 晚於決策日的版本；免費的 Yahoo 路徑拿不到同一期的多個版本。

設了 `FMP_API_KEY` 時，預設（`BACKTEST_FUNDAMENTALS_SOURCE=auto`）改向 Financial Modeling Prep 要損益表與資產負債表。沒有金鑰就走 Yahoo，不會因此壞掉。`BACKTEST_FUNDAMENTALS_SOURCE=yahoo` 可以強制走 Yahoo。明確設成 `fmp` 但沒有金鑰會報錯。FMP 的標準化財報也可能被供應商就地改寫，仍然不是機構級 point-in-time。台股與非美股代碼是原樣送出，FMP 沒有資料時，auto 會再試 Yahoo。

台股若 Yahoo 財報是空的，該列就是空值，篩選會跳過，不會補一個假數字。

價格因子在時點模式下用決策日往前約 12 個月（365 個日曆天）的收盤價。決策日之後的價格不會進動能或波動。

互動介面可以改選「當下快照」。那是 Yahoo `ticker.info` 的今天數值，畫面會標明不能拿來做歷史驗證。

成功取回的財報表寫進 `data_cache/fundamentals/`（Parquet）。跟股價共用 `BACKTEST_DATA_CACHE` 與 `BACKTEST_DATA_CACHE_DIR`。預設最多沿用 1 天，用 `BACKTEST_FUNDAMENTALS_CACHE_TTL_DAYS` 改。

## 成交與成本

策略在第 t 根 K 線收盤時看到訊號。回測在第 t+1 根的**開盤價**成交，不會用第 t 根的收盤價成交。第 t 根收盤時，帳戶仍用上一筆成交之後的部位，以收盤價盯市。

最後一根 K 線上的訊號沒有下一根開盤可以成交，引擎不會改在當根收盤強制平倉，部位留到最後一根收盤價盯市。樣本第一根也不會成交：沒有更早的收盤訊號落在這段資料裡。

滑價從開盤價往不利方向偏移：買入乘 `(1 + slippage)`，賣出乘 `(1 - slippage)`。預設 5 bps（`slippage=0.0005`）。`slippage_bps` 用 bps 覆寫，例如 `slippage_bps=5`。

市場判定：代碼以 `.TW` 或 `.TWO` 結尾為台股，其餘（含 `AAPL`、`BTC-USD`）為美股。`run(..., market="US"|"TW"|"FUTURES")` 可以覆寫後綴。期貨不會從代碼猜。

美股與其他非台股（預設）：

- 手續費是成交金額的比例，買賣都收，預設 10 bps（`commission=0.001`）。`commission_bps` 可以改用 bps。另外可加每筆固定金額 `commission_flat`，預設 0。
- 費用從成交金額算，不是從下單前的現金算。買進時把現金全部用完：名目金額 `= (現金 - 固定費用) / (1 + 費率)`。
- 沒有模擬股息，也不會讀資料裡的股息欄。`dividend_withholding` 預設 0.30，只記在成本設定上；以後若把股息放進回測，美股預扣用這個比例。現在改這個數字不會改變淨值。

台股與台股 ETF（`.TW` / `.TWO`）：

- 上面的美股手續費滑桿與 `commission` 不套用。滑價仍套用。
- 買賣都收券商手續費，預設法定上限 0.1425%，單筆最低 20（與價格同一幣別；台股報價就是新台幣 20 元）。這不是電子券商折扣。六折要自己設 `brokerage_rate=0.001425 * 0.6`。
- 賣出另收證交稅，稅基是成交金額。一般股票 0.3%。去掉後綴後以 `00` 開頭的代碼（`0050`、`00631L` 這類）用 0.1%。這是代碼慣例，不是交易所的證券分類；債券 ETF 實務上常停徵，這裡仍會收到 0.1%，需要的話傳 `sell_tax_rate=0`。
- 當沖證交稅減半（0.15%）沒有套用。日線、下一根開盤成交，湊不出日內現股當沖。
- 股數可以是碎股，不會自動湊成 1000 股的整張。

`market="FUTURES"` 仍用美股那套手續費與滑價，而且是全額付現，不是保證金。`contract_multiplier` 把報價換成每口名目（損益 `= 口數 × 價格 × 乘數`）。`min_lot` 會把口數無條件捨去到整數倍，剩下的現金留著。兩個都不設時乘數是 1、也不取整，回測不會因為選了 FUTURES 就停住。

多檔標的的做法與以前相同：每檔用等分資金各自全進全出，淨值再相加。引擎不換匯。美股和台股放在同一個組合裡時，數字會直接相加。

跑單元測試（不需要金鑰）：

```bash
pip install -r requirements-dev.txt
pytest
```

## 免責聲明

本專案僅供教育與研究用途，所有回測結果基於歷史資料，**不構成任何投資建議**。歷史績效不代表未來表現，實際交易存在滑價、流動性、稅務等成本，請自行評估風險。

---

_預設資料來源：Yahoo Finance_
