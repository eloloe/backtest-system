# 量化金融回測系統 (Quant Backtesting System)

一個以 **Streamlit** 打造的互動式量化策略回測平台，整合基本面篩選、多因子評分、技術策略回測與完整績效／風險分析，可一站式驗證選股與交易策略。

**線上 Demo**：https://backtest-system-gfba5qpzyz3hzc6jbsuj2u.streamlit.app/

---

## 功能特色

- **商品池設定**：支援美股、ETF、指數、台股（`.TW`）、加密貨幣（`BTC-USD` 等）。預設資料來源為 Yahoo Finance，可改接 Tiingo、Polygon 或永豐 Shioaji（見下方）。
- **基本面濾網**：以 PE、ROE、營收成長率、市值等硬門檻篩選股票池。歷史回測用起始日當時已公開的財報，不用今天的快照。
- **多因子評分系統**：ROE / PE / PB / 營收成長 / 盈餘成長 / 動能 / 波動率等因子，可自由開關與調整權重，支援 Z-Score 與百分位排名兩種標準化，選出綜合分數 Top N。
- **八種交易策略**：
  - 買進持有 (Buy & Hold)
  - 均線交叉 (MA Cross，SMA / EMA)
  - RSI 策略
  - 布林通道 (Bollinger Bands)
  - MACD 策略
  - 趨勢停損 (Trend + Stop)：慢均線之上才持有，再用 ATR 或百分比停損把虧損砍掉
  - 橫截面多因子：再平衡日用當時已公開的財報和動能排名，等權或波動倒數持有前 N 名
  - 核心衛星：一開始按固定比例切資金（例如 70/30），兩袖之後各自漲跌
- **事件驅動回測引擎**：訊號在收盤決定、下一根開盤成交。美股用手續費與滑價，台股用券商手續費與賣出證交稅。單一名稱的策略仍是等權重、各檔全進全出。橫截面改成一個現金帳戶、按目標權重調倉；核心衛星只在一開始把資金切開。
- **Walk-forward**：滾動的訓練窗、禁運、測試窗，樣本外淨值另計。末端可留一段鎖定的 holdout，預設不打開。一般的整段回測仍是預設。
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
| `app.py` | Streamlit 畫面。控件和圖留在這裡，回測呼叫 `runner` |
| `cli.py` | 同一條回測的命令列。`python cli.py` 或 `python -m cli` |
| `runner.py` | 讀 YAML、跑回測或 walk-forward、寫出淨值／回撤圖和指標 |
| `examples/` | 三個策略的範例設定。沒有金鑰 |
| `books.py` | 等權單名帳戶、目標權重調倉、兩袖相加 |
| `data_fetcher.py` | 依設定的 provider 取得 OHLCV |
| `providers/` | 價格來源（Yahoo / Tiingo / Polygon / Shioaji）與本機 Parquet 快取 |
| `fundamentals.py` | 基本面入口。當下快照與 point-in-time 都從這裡進 |
| `fundamentals_pit.py` | 依 `as_of` 用已公開財報與當時股價算 PE、PB、ROE 等 |
| `screener.py` | 硬門檻篩選與因子評分整合 |
| `factor_engine.py` | 因子註冊表、價格因子計算、標準化與綜合評分 |
| `strategies.py` | 訊號，以及橫截面／核心衛星 |
| `backtester.py` | 單一名稱的回測（下一根開盤成交、美股／台股成本） |
| `walk_forward.py` | 滾動訓練／禁運／測試、鎖定 holdout、樣本外拼接 |
| `metrics.py` | 績效與風險指標計算 |


---

## 資料來源

股價走 `providers/`。用環境變數 `BACKTEST_DATA_PROVIDER` 選擇，預設 `yahoo`，不需要金鑰就能跑 Demo。

| 名稱 | 金鑰 | 涵蓋範圍 | 價格調整 |
|------|------|----------|----------|
| `yahoo`（預設） | 無 | 沿用 Yahoo 代碼：美股、ETF、指數、台股（`.TW`）、加密貨幣（`BTC-USD`） | `auto_adjust=True`，除權息 |
| `tiingo` | `TIINGO_API_KEY` | 美股、ETF 日線 | 使用 `adj*` 欄位（除權息） |
| `polygon` | `POLYGON_API_KEY` | 美股、ETF 日線 | `adjusted=true`，只還原拆股，不含股息 |
| `shioaji` | `SHIOAJI_API_KEY`、`SHIOAJI_SECRET_KEY` | 台股、台股 ETF、台指期 | 交易所成交價，不還原除權息。預設模擬環境 |

切換範例：

```bash
export BACKTEST_DATA_PROVIDER=tiingo
export TIINGO_API_KEY=你的金鑰
streamlit run app.py
```

Polygon 把 `BACKTEST_DATA_PROVIDER` 設成 `polygon`，金鑰變數是 `POLYGON_API_KEY`。選了需要金鑰的來源卻沒設時會直接報錯，不會悄悄退回 Yahoo，專案裡也不會放任何金鑰。

成功取回的日線會寫進 `data_cache/`（Parquet，已列入 `.gitignore`）。鍵是 provider、代碼、起始日與結束日。關掉快取：`BACKTEST_DATA_CACHE=0`。換目錄：`BACKTEST_DATA_CACHE_DIR`。

`end` 不含當日，跟 `yfinance` 的 `history(end=...)` 一樣。

### 永豐 Shioaji（台股、台股 ETF、台指期）

美股、其他市場和加密貨幣不要走這條。`AAPL`、`BTC-USD` 會直接報錯，請改 `yahoo`、`tiingo` 或 `polygon`。一次執行只有一個 `BACKTEST_DATA_PROVIDER`，不會在同一條請求裡把台股分給永豐、美股分給 Tiingo。兩邊都要時，分兩次跑，或在呼叫端換 provider。

```bash
pip install -r requirements-shioaji.txt
export BACKTEST_DATA_PROVIDER=shioaji
export SHIOAJI_API_KEY=你的 API Key
export SHIOAJI_SECRET_KEY=你的 Secret Key
streamlit run app.py
```

`shioaji` 不在 `requirements.txt` 裡，沒裝也不影響原本的 Demo。金鑰只從這兩個環境變數讀，傳給 `api.login(api_key=..., secret_key=...)`。缺一就丟 `MissingApiKeyError`。不要把金鑰或帳號寫進 repo。

預設是模擬環境：`sj.Shioaji(simulation=True)`。這裡只抓 K 線，不啟用憑證，也不下單。要連正式環境再設 `SHIOAJI_SIMULATION=0`。Shioaji 得連得到永豐，多半要在自己的電腦上跑（例如 Mac）；雲端 Demo 通常沒有這條線路。

| 你傳的代碼 | 向 Shioaji 查 |
|------------|----------------|
| `2330.TW`、`2330.TWO`、`2330` | `2330` |
| `0050.TW`、`0050` | `0050` |
| `TXF` | `TXFR1`（台指期近月連續，結算日換月） |
| `TXFR1`、`TXFR2`、`TXFH6` | 原樣。`TXFR2` 是次月；月份代碼不會被改成近月 |

日線不是另一支 API。`kbars` 是一分鐘線，一次最多 30 個日曆日，長區間會拆開抓再合成。`end` 一樣不含當日。股票用日曆日。台指期把 15:00 以後算到下一個交易日，週五夜盤到週六凌晨算下週一；國定假日沒有再往後順延。價格是交易所成交價，不還原除權息。股票成交量是張，期貨是口，跟 Yahoo 台股用「股」計的成交量不同。

股票分鐘線大約從 2020-03-02 起，期貨大約從 2020-03-22 起。抓成功的日線一樣寫進 `data_cache/`。

目前的限制：

- Yahoo 適合示範。調整方式與服務穩定性都不該當成研究級資料。
- Tiingo 與 Polygon 只接美股／ETF 日線，代碼原樣送出。台股與台指期用 Shioaji。加密貨幣仍走 Yahoo。
- Shioaji 不還原除權息，股票成交量是張。和 Yahoo 的還原股價、成交股數不會對齊。
- Polygon 沒有做股息調整，收盤價不會和 Yahoo／Tiingo 的總報酬價格對齊。
- 歷史篩選用起始日的 point-in-time 財報，不是 Yahoo 今天的快照。範圍與做不到的地方見下一節。

## 基本面（point-in-time）

歷史回測的篩選與因子評分對齊回測起始日。PE、PB、ROE、營收成長、盈餘成長、市值只用該日以前看得到的財報，再搭配該日（或之前最後一個交易日）的收盤價。市值用普通股股數；沒有才退回基本或稀釋股數。動能與波動率是往回大約十二個月、結束在起始日收盤的價格；不會把回測區間後面的漲跌算進選股。畫面上的濾網只在起始日選一次，沒有沿著樣本往前換股。橫截面策略是例外，它在每個再平衡日各自重排，見下方「橫截面多因子」。

預設不需要金鑰。季報與年報來自 Yahoo 的期間財報。Yahoo 這條免費序列很短，季報大約五季、年報大約四年。更早的日期若對不上財報，ROE、PB、營收成長就是空值，硬門檻會把缺資料的標的拿掉，不會補一個數。盈餘若能對上法說日，PE 與盈餘成長可以沿用那串已公布的 EPS，公布日就是看得到的日期。抓不到法說日或申報日時，假設報表在會計期間結束後 60 個日曆天才公開。這個天數是假設，不是公司實際的申報日。要改的話設 `BACKTEST_FUNDAMENTALS_LAG_DAYS`（非負整數）。已知的法說日或申報日若比這個落後還晚，就等到那天，不會提前用落後天數把報表放出來。

設了 `FMP_API_KEY` 時改向 Financial Modeling Prep 拿季報，可見日用 `filingDate`（沒有則用舊欄位 `fillingDate`，再沒有才用 `acceptedDate` 的日期）。資產負債表與損益表的申報日取較晚的那天。沒有金鑰、金鑰是空白、或請求失敗，會回到 Yahoo，不會因此不能跑。

抓回來的財報面板寫進 `data_cache/fundamentals/`，跟股價同一套開關（`BACKTEST_DATA_CACHE`、`BACKTEST_DATA_CACHE_DIR`）。空結果不寫入。

畫面上可以改勾「Yahoo 今天的快照」。那是 `ticker.info` 的當下數字，畫面會標明它有前視，不能當成歷史結果。

做不到的事：

- 這不是 Compustat、FactSet 或 TEJ，也沒有完整的 as-reported 修正史。Yahoo 或 FMP 若一季只給一個數字，那是他們現在畫面上的版本；後來的重編可能已經混在原始公布日裡。面板裡若同一季有兩版而且各有申報日，`as_of` 只會用申報日已經到的那一版。
- 估值用的收盤價走目前的價格 provider。Yahoo 預設含除權息，配息會把歷史股價往下修，PE、PB、市值會偏一點。Shioaji 的成交價不還原，和 Yahoo 財報的股數基礎不一定對得起來。
- 盈餘小於或等於零時 PE 留空。
- 台股在 Yahoo 上常常沒有夠用的期間財報。缺了就是空值，篩選跳過，不估計。
- 申報日當天就當成全天可用。若檔案是收盤後才上去，日線會早幾小時。
- 成交與成本沒有因為這一步改動。樣本外的滾動驗證見下方 Walk-forward。

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

單一名稱的策略仍是每檔等分資金、各自全進全出，淨值再相加。橫截面是一個現金帳戶按目標權重調倉，核心衛星只在一開始切資金，見下方「三種新策略」。引擎不換匯。美股和台股放在同一個組合裡時，數字會直接相加。

跑單元測試（不需要金鑰，也不必安裝 `shioaji`）：

```bash
pip install -r requirements-dev.txt
pytest
```

## Walk-forward

一般回測用同一組參數跑完整段 `[start, end)`，樣本內好看的數字會把研究品質說得太滿。側邊欄「Walk-forward 驗證」預設關掉，Demo 還是原來的整段回測。打開之後，畫面只留滾動測試的結果，不再畫那條整段淨值。

區間跟股價一樣是半開的，結束日當天不包含。每一折：

1. **訓練窗**，預設 24 個月。格點若多於一組，只在這段裡選參。
2. **禁運**，訓練結束之後的一段日曆日，預設 21 天，至少 1 天。這幾天不進選參的目標函數，也不計入該折的測試損益。用日曆日，不是交易日：1 月 1 日加 21 天就是 1 月 22 日，中間的週末照算。
3. **測試窗**，預設 6 個月。參數已經定了，只在這段上評。

下一折的起點往後移一個測試窗，所以測試窗首尾相接、不共用任何一天。排不進完整測試窗的尾巴丟掉，不會縮短來硬湊。後一折的訓練會蓋到較早的測試期，那是後來已經發生的資料。同一折的訓練、禁運、測試三段彼此不重疊。因為禁運比測試窗短，下一折的禁運有時會落在上一折測試窗的尾巴；那段仍只算上一折的樣本外，下一折選參不會用到。

選參指標是扣過成本的 Sharpe，走現有的 `Backtester` 跟 `metrics.calculate_metrics`。平手留格點裡較前面的那組，不看測試段。各折測試窗的報酬接成一條樣本外淨值，再算一次整體指標。部位不跨禁運，也不帶進下一折，每個測試窗從現金重新開始。禁運那幾天不補成零報酬。指標若要暖機，只用窗口起點之前的價格；選參看到的價格停在訓練窗結束之前。

有參數的策略用下面的小格點，不是側邊欄上那一個滑桿。買進持有沒有格點，每一折直接評。

| 策略 | 格點 |
|------|------|
| 均線交叉 | SMA (10, 30)、(20, 60)、(50, 100)，EMA (20, 60) |
| RSI | 週期／超賣／超買：(14, 30, 70)、(14, 20, 80)、(7, 30, 70)、(21, 30, 70) |
| 布林通道 | 週期 20 搭配 2.0、2.5、1.5 倍標準差，以及週期 10 搭配 2.0 |
| MACD | 快／慢／信號線：(12, 26, 9)、(8, 21, 9)、(12, 26, 5)、(5, 35, 5) |
| 趨勢停損 | 均線 50 或 100；停損是 8% 或 10%，或 ATR 2.5／3 倍 |
| 橫截面多因子 | 前 2 或 3 名；等權或波動倒數；動能用 63 或 126 日。比單次回測短，因為每一折只看得到窗口前的暖機 |
| 核心衛星 | 權重固定 70/30，不搜尋。格點只換核心是買進持有還是慢趨勢、衛星是趨勢停損還是均線 |

格點很小，只是不讓單一一組參數代表整個樣本。多組一起試，Sharpe 還是會偏樂觀。Deflated Sharpe、Holm 這類多重檢定校正不在這次範圍。橫截面在 walk-forward 裡仍用 `fetch_fundamentals_pit`，不用 Yahoo 今天的快照。

### 鎖定的 holdout

「末端鎖定 holdout（月）」從結束日往前留一段，預設 12 個月，可以設成 0。留下的時候，這一段不進任何一折的訓練或測試。預設不計算。

要看這一段，得另外勾「打開鎖定的 holdout」，或在程式裡呼叫 `evaluate_locked_holdout(..., open_holdout=True)`。沒有 `open_holdout=True` 會直接拒絕。打開之後用**最後一折實際用過的參數**，原樣評在 holdout 上，不再搜尋。結果單獨列，有自己的淨值與回撤，不併進樣本外曲線。不要拿 holdout 的好壞回頭改格點或視窗；改了，這段就不再是留白。

設成 0 表示不留這一段，尾端只要夠長就會變成一般的測試折。

### 選股時點

有開基本面篩選或因子評分時，每一折的 `as_of` 是**該折訓練窗的第一天**。訓練和該折測試用同一份名單，走 `fetch_fundamentals_pit`，不呼叫 Yahoo 今天的快照。側邊欄那個前視選項在這個模式裡不起作用。動能與波動率也停在同一個 `as_of`。Holdout 若打開，名單改以 holdout 起點重算一次，參數仍是最後一折鎖定的那組。某一折篩完沒有標的，該折測試段以現金計，不另挑一組參數充數。橫截面不走這層名單：它在測試窗裡的每個再平衡日自己呼叫 point-in-time。

跑完會畫樣本外的淨值與回撤。Holdout 打開時另外兩張。

```bash
streamlit run app.py
```

左側勾「Walk-forward 驗證」，再按開始回測。只跑分割與選參的測試、不走網路：

```bash
pytest tests/test_walk_forward.py
```

## 命令列

Streamlit 跟命令列走同一條 `runner.run_research`。畫面只負責控件和圖。`strategies.py`、`factor_engine.py`、`books.py`、`runner.py` 不 import Streamlit。今天的 Yahoo 快照還留在 `fundamentals.py`，給畫面上那個前視選項；命令列的歷史選股不呼叫它。

```bash
pip install -r requirements.txt
python cli.py --config examples/trend_stop.yaml
python -m cli --config examples/cross_section.yaml --start 2018-01-01 --end 2024-01-01
```

金鑰不要寫進 YAML。價格來源用 `provider:`（或環境變數 `BACKTEST_DATA_PROVIDER`）。Tiingo、Polygon、Shioaji、FMP 的 key 仍放在原來的環境變數。設定檔裡若出現 `api_key`、`secret` 這類欄位，讀取時會直接拒絕。

跑完會在 `output_dir` 寫：

- `equity.html`、`drawdown.html`：淨值和回撤。Plotly 的 HTML，用瀏覽器打開。
- `equity.csv`
- `metrics.csv`、`metrics.json`：跟終端機印出來的是同一張表
- 有成交時多一個 `trades.csv`。橫截面再多 `holdings.csv`，每一列是某個再平衡日當時排得進來的標的
- 開了 walk-forward 再多 `folds.csv`。若同時 `--open-holdout`，另寫 holdout 的兩張圖

`output/` 已在 `.gitignore`。

```bash
python cli.py --config examples/trend_stop.yaml --walk-forward
python cli.py --config examples/core_satellite.yaml --no-walk-forward
```

Walk-forward 用上面那張小格點，不用 YAML `params` 裡的那一組。`--walk-forward` 和 `--no-walk-forward` 蓋過檔案裡的開關。`--open-holdout` 才會評末端那段，而且不拿它選參。

範例：

```yaml
strategy: trend_stop
tickers: [SPY, QQQ]
benchmark: SPY
start: 2016-01-01
end: 2024-01-01          # 不含這一天
initial_capital: 100000
commission: 0.001
slippage: 0.0005
risk_free_rate: 0.04
provider: yahoo          # 沒有金鑰
output_dir: output/trend_stop
warmup_bars: 120         # 只給均線暖機。start 之前的訊號會清成空手
walk_forward: false

params:
  trend_window: 100
  ma_type: SMA
  stop_mode: atr         # atr、percent，或 tighter（兩條裡較高的停損）
  atr_period: 14
  atr_mult: 3.0
  stop_pct: 0.08
```

另外兩份在 `examples/cross_section.yaml` 和 `examples/core_satellite.yaml`。

## 三種新策略

三個都沿用下一根開盤成交，和現有的美股手續費／滑價、台股手續費／證交稅。沒有盤中觸價。

### 趨勢停損

收盤在慢均線之上才做多，停損從這一段的最高收盤往下算。`atr` 是最高收盤減 ATR 倍數，`percent` 是最高收盤乘上 `(1 - stop_pct)`，`tighter` 取兩者裡較高的那條。收盤跌破停損，或收到均線下，訊號改成空手；賣出在下一根開盤。

停損出場時若價格還在均線上，不會下一根就買回。要先收到均線下面，再重新站上，才算下一段。樣本一開始若已經在均線上，允許進場。`warmup_bars` 只讓均線在 `start` 當天算得動，不會把暖機期間的部位帶進區間。

### 橫截面多因子

小宇宙，在再平衡日收盤排名，下一根開盤調倉。再平衡日是「下一個交易日已經落到下個月」的那個收盤。樣本停在月中，不會把最後那根當成月末。`rebalance_months: 3` 是每第三個這樣的日子，從窗口裡的第一個算起。

當天排得進來的東西只有：

- 收盤，而且是 `as_of` 當天或更早。動能預設是 `t-21` 的收盤相對再往前 252 個交易日，最近一個月不計。
- 呼叫端給的財報。實盤路徑是 `fetch_fundamentals_pit`，可見日規則跟基本面那一節一樣。本益比要大於 0，價值用 `1/PE`。
- 波動倒數用的波動，也是停在 `as_of` 的收盤。年化只為了好讀，權重看的是相對大小。

缺了所需的那一腿就這期不排，不會拿橫截面的中位數去補。前 N 名等權，或按波動倒數把權重加總成 1。這不是優化器：不加槓桿、不看共變異。某名沒有可用的波動時，該期退回等權。兩次再平衡之間股數不變，權重隨價格漂。`holdings.csv` 的 `as_of` 就是上面這個收盤日。

### 核心衛星

`core_weight` 和 `satellite_weight` 要加總為 1。資金在第一天按這個比例切開，之後各算各的，不每期拉回 70/30，也沒有均值變異或風險平價之外的求解。波動倒數只出現在衛星若選了橫截面、而且 `weighting: risk` 的時候，袖與袖之間不動。

核心只能是買進持有，或同一套趨勢停損。衛星可以是趨勢停損、原本的四個技術策略，或橫截面。`core_symbol` 有填的話，核心只交易那一檔，而且這檔要在價格裡；沒填就跟衛星用同一份清單、等權。`examples/core_satellite.yaml` 是核心 SPY 買進持有 70%、衛星在股票池上做趨勢停損 30%。

Walk-forward 的核心衛星格點把權重釘在 70/30，只換袖的做法，不搜尋比例。

## 免責聲明

本專案僅供教育與研究用途，所有回測結果基於歷史資料，**不構成任何投資建議**。歷史績效不代表未來表現，實際交易存在滑價、流動性、稅務等成本，請自行評估風險。

---

_預設資料來源：Yahoo Finance_
