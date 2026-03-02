"""
デバッグ分析：シグナル条件ごとに何件が通過/脱落しているか可視化
"""
import yfinance as yf
import pandas as pd
from src import config

def _calc_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

print(f"対象銘柄数: {len(config.TARGET_TICKERS)}")
print("データ取得中...")

data = yf.download(
    config.TARGET_TICKERS,
    period="6mo",
    group_by='ticker',
    progress=False,
    auto_adjust=False
)

total = 0
fail_no_data = 0
fail_short_history = 0
pass_all = 0

# 条件ごとの脱落カウント
cond1_fail = 0  # 終値 > SMA200 を満たさない
cond2_fail = 0  # 終値 < SMA5 を満たさない
cond3_fail = 0  # RSI(2) < 10 を満たさない
cond12_pass = 0 # 条件1,2は通過

# 条件1&2を通過した銘柄のRSI分布
rsi_values = []

for ticker in config.TARGET_TICKERS:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.levels[0]:
                fail_no_data += 1
                continue
            df = data[ticker].copy()
        else:
            df = data.copy()

        df.dropna(subset=['Close'], inplace=True)
        if len(df) < config.P_SMA_LONG:
            fail_short_history += 1
            continue

        total += 1
        close = df['Close']
        sma5 = close.rolling(config.P_SMA_SHORT).mean().iloc[-1]
        sma200 = close.rolling(config.P_SMA_LONG).mean().iloc[-1]
        rsi2 = _calc_rsi(close, config.P_RSI).iloc[-1]
        c_price = close.iloc[-1]

        c1 = c_price > sma200
        c2 = c_price < sma5
        c3 = rsi2 < config.RSI_THRESHOLD

        if not c1:
            cond1_fail += 1
        elif not c2:
            cond2_fail += 1
        else:
            cond12_pass += 1
            rsi_values.append((ticker, rsi2))
            if not c3:
                cond3_fail += 1
            else:
                pass_all += 1

    except Exception as e:
        fail_no_data += 1

print()
print("=" * 50)
print("【シグナル条件別 脱落分析】")
print("=" * 50)
print(f"  データ取得失敗 / 上場廃止等  : {fail_no_data}件")
print(f"  データ不足（6ヶ月未満）      : {fail_short_history}件")
print(f"  分析対象                      : {total}件")
print()
print(f"  ❌ 条件1 失敗（終値 < SMA200） : {cond1_fail}件  ← 長期下降トレンド")
print(f"  ❌ 条件2 失敗（終値 > SMA5）  : {cond2_fail}件  ← 短期押し目なし")
print(f"  ✅ 条件1,2 通過               : {cond12_pass}件")
print()
if rsi_values:
    rsi_df = pd.DataFrame(rsi_values, columns=['ticker', 'rsi2']).sort_values('rsi2')
    print(f"  ❌ 条件3 失敗（RSI(2) >= 10）: {cond3_fail}件")
    print(f"  ✅ 全条件通過（シグナル）    : {pass_all}件")
    print()
    print("【条件1&2通過銘柄 RSI(2)分布】")
    bins = [0, 10, 20, 30, 50, 100]
    labels = ["<10(シグナル)", "10-20", "20-30", "30-50", "50+"]
    rsi_series = rsi_df['rsi2']
    for i in range(len(bins)-1):
        cnt = ((rsi_series >= bins[i]) & (rsi_series < bins[i+1])).sum()
        print(f"  RSI {labels[i]:15s}: {cnt}件")
    print()
    print("【RSI(2)が低い上位10銘柄（最もシグナルに近い）】")
    print(rsi_df.head(10).to_string(index=False))
else:
    print("  条件1&2を通過した銘柄がないため、RSI分析不可")
print("=" * 50)
