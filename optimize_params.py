"""
バックテスト最適化 — 指標・パラメータの網羅的探索
戦略パターン A〜D を横断的に評価し、利益を最大化するパラメータセットを特定する。
"""
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import itertools
import time
from src import config

# 重複除去
TICKERS = list(dict.fromkeys(config.TARGET_TICKERS))

# ============================================================
# 指標計算
# ============================================================
def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_zscore(series: pd.Series, period: int) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - sma) / std

def calc_daily_return_zscore(series: pd.Series, period: int) -> pd.Series:
    daily_returns = series.pct_change()
    mean_return = daily_returns.rolling(period).mean()
    std_return = daily_returns.rolling(period).std()
    return (daily_returns - mean_return) / std_return

def calc_historical_volatility(series: pd.Series, period: int) -> pd.Series:
    daily_returns = series.pct_change()
    # 年率換算なしの単純な期間HV（単位を揃えるため）
    return daily_returns.rolling(period).std() * np.sqrt(252)


# ============================================================
# データ取得（1回だけ実行）
# ============================================================
def fetch_data():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=500)  # 余裕をもって取得

    print(f"Fetching data from {start_date} to {end_date} for {len(TICKERS)} tickers...")
    data = yf.download(TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=False)

    try:
        data.index = data.index.tz_convert('Asia/Tokyo')
    except Exception:
        data.index = data.index.tz_localize('Asia/Tokyo')

    eval_start = pd.to_datetime(end_date - datetime.timedelta(days=90)).tz_localize('Asia/Tokyo')
    dates = sorted([d for d in data.index.unique() if d >= eval_start])

    # 個別銘柄のDataFrameに分解（1回だけ）
    ticker_dfs = {}
    for ticker in TICKERS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.levels[0]:
                    continue
                df = data[ticker].copy()
            else:
                df = data.copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) >= 200:  # 最低200日
                ticker_dfs[ticker] = df
        except Exception:
            pass

    print(f"Usable tickers: {len(ticker_dfs)}")
    return ticker_dfs, dates


# ============================================================
# 共通バックテストエンジン
# ============================================================
def run_backtest(ticker_indicators, dates, signal_fn, exit_fn, params):
    """
    ticker_indicators: {ticker: DataFrame with indicators}
    signal_fn(row, params) -> bool  : 買いシグナル判定
    exit_fn(row, pos, today, params) -> bool : 売りシグナル判定
    """
    initial_capital = 1_000_000
    capital = initial_capital
    positions = {}
    trade_history = []
    max_positions = params.get('max_positions', 5)
    max_position_size = initial_capital * 0.20

    for idx, today in enumerate(dates):
        # --- 売却処理 ---
        if idx > 0:
            yesterday = dates[idx - 1]
            sold = []
            for ticker, pos in positions.items():
                df = ticker_indicators[ticker]
                if yesterday not in df.index or today not in df.index:
                    continue
                y_row = df.loc[yesterday]
                t_open = df.loc[today, 'Open']
                if pd.isna(t_open):
                    continue

                days_held = (today - pos['entry_date']).days
                if exit_fn(y_row, pos, days_held, t_open, params):
                    profit = (t_open - pos['entry_price']) * pos['shares']
                    capital += t_open * pos['shares']
                    trade_history.append({
                        'profit': profit,
                        'return_pct': (t_open / pos['entry_price'] - 1) * 100
                    })
                    sold.append(ticker)
            for t in sold:
                del positions[t]

        # --- 購入処理 ---
        if idx == 0:
            continue
        yesterday = dates[idx - 1]
        available_cash = capital
        candidates = []

        for ticker, df in ticker_indicators.items():
            if ticker in positions:
                continue
            if yesterday not in df.index or today not in df.index:
                continue
            y_row = df.loc[yesterday]
            if signal_fn(y_row, params):
                t_open = df.loc[today, 'Open']
                if pd.isna(t_open):
                    continue
                # スコア: RSIベースまたはシンプルな固定値
                rsi_val = y_row.get('RSI', 50)
                rsi_thresh = params.get('rsi_thresh', 30)
                score = max(0, rsi_thresh - rsi_val) if not pd.isna(rsi_val) else 0
                candidates.append((ticker, t_open, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for ticker, t_open, _ in candidates:
            if len(positions) >= max_positions:
                break
            max_s = int(max_position_size // (t_open * 100)) * 100
            cash_s = int(available_cash // (t_open * 100)) * 100
            shares = min(max_s, cash_s)
            if shares >= 100:
                cost = shares * t_open
                capital -= cost
                available_cash -= cost
                positions[ticker] = {
                    'shares': shares,
                    'entry_price': t_open,
                    'entry_date': today
                }

    # 全決済
    if dates:
        last = dates[-1]
        for ticker, pos in list(positions.items()):
            df = ticker_indicators[ticker]
            if last in df.index:
                c = df.loc[last, 'Close']
                profit = (c - pos['entry_price']) * pos['shares']
                capital += c * pos['shares']
                trade_history.append({
                    'profit': profit,
                    'return_pct': (c / pos['entry_price'] - 1) * 100
                })

    total_return = ((capital / initial_capital) - 1) * 100
    n_trades = len(trade_history)
    wins = [t for t in trade_history if t['profit'] > 0]
    win_rate = (len(wins) / n_trades * 100) if n_trades else 0
    avg_return = np.mean([t['return_pct'] for t in trade_history]) if trade_history else 0

    return {
        'return': round(total_return, 2),
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
    }


# ============================================================
# パターン A: RSI + SMA（RSI期間も探索）
# ============================================================
def strategy_a_prep(ticker_dfs, rsi_period, sma_short, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['RSI'] = calc_rsi(d['Close'], rsi_period)
        d['SMA_SHORT'] = d['Close'].rolling(sma_short).mean()
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_a_signal(row, params):
    return (row['Close'] > row['SMA_LONG'] and
            row['Close'] < row['SMA_SHORT'] and
            row['RSI'] < params['rsi_thresh'])

def strategy_a_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    return (row['Close'] > row['SMA_SHORT'] or
            days_held >= params['hold_days'] or
            t_open <= sl_price)

STRATEGY_A_GRID = {
    'rsi_period': [2, 3, 5, 7, 14],
    'rsi_thresh': [10, 15, 20, 25, 30],
    'sma_short': [3, 5, 7, 10],
    'sma_long': [100, 150, 200],
    'stop_loss': [0.03, 0.05, 0.08],
    'hold_days': [5, 10, 15],
}


# ============================================================
# パターン B: MACD + トレンドフィルター
# ============================================================
def strategy_b_prep(ticker_dfs, macd_fast, macd_slow, macd_signal, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['MACD'], d['MACD_SIGNAL'], d['MACD_HIST'] = calc_macd(d['Close'], macd_fast, macd_slow, macd_signal)
        d['MACD_HIST_PREV'] = d['MACD_HIST'].shift(1)
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_b_signal(row, params):
    # MACDヒストグラムがマイナス→プラスに転換 & トレンドSMAより上
    return (row.get('MACD_HIST', 0) > 0 and
            row.get('MACD_HIST_PREV', 0) <= 0 and
            row['Close'] > row['SMA_LONG'])

def strategy_b_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    # MACDヒストグラムがプラス→マイナスに転換 or 損切り or 日数
    macd_exit = (row.get('MACD_HIST', 0) < 0 and row.get('MACD_HIST_PREV', 0) >= 0)
    return macd_exit or days_held >= 15 or t_open <= sl_price

STRATEGY_B_GRID = {
    'macd_fast': [8, 12],
    'macd_slow': [21, 26],
    'macd_signal': [9],
    'sma_long': [100, 150, 200],
    'stop_loss': [0.03, 0.05, 0.08],
}


# ============================================================
# パターン C: ボリンジャーバンド + RSI
# ============================================================
def strategy_c_prep(ticker_dfs, bb_period, bb_std, rsi_period, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['BB_UPPER'], d['BB_MID'], d['BB_LOWER'] = calc_bollinger(d['Close'], bb_period, bb_std)
        d['RSI'] = calc_rsi(d['Close'], rsi_period)
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_c_signal(row, params):
    # 下バンドタッチ + RSI売られすぎ + 上昇トレンド
    return (row['Close'] <= row['BB_LOWER'] and
            row['RSI'] < params['rsi_thresh'] and
            row['Close'] > row['SMA_LONG'])

def strategy_c_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    # 中心線（BB_MID）回帰 or 損切り or 日数
    return (row['Close'] >= row['BB_MID'] or
            days_held >= 15 or
            t_open <= sl_price)

STRATEGY_C_GRID = {
    'bb_period': [10, 20],
    'bb_std': [1.5, 2.0, 2.5],
    'rsi_period': [2, 5, 14],
    'rsi_thresh': [20, 30, 40],
    'sma_long': [100, 200],
    'stop_loss': [0.03, 0.05, 0.08],
}


# ============================================================
# パターン D: ATR動的損切り + RSI
# ============================================================
def strategy_d_prep(ticker_dfs, atr_period, rsi_period, sma_short, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['ATR'] = calc_atr(d['High'], d['Low'], d['Close'], atr_period)
        d['RSI'] = calc_rsi(d['Close'], rsi_period)
        d['SMA_SHORT'] = d['Close'].rolling(sma_short).mean()
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_d_signal(row, params):
    return (row['Close'] > row['SMA_LONG'] and
            row['Close'] < row['SMA_SHORT'] and
            row['RSI'] < params['rsi_thresh'])

def strategy_d_exit(row, pos, days_held, t_open, params):
    # ATRベース動的損切り
    atr_val = row.get('ATR', 0)
    atr_mult = params.get('atr_mult', 2.0)
    sl_price = pos['entry_price'] - atr_val * atr_mult
    return (row['Close'] > row['SMA_SHORT'] or
            days_held >= 15 or
            t_open <= sl_price)

STRATEGY_D_GRID = {
    'atr_period': [10, 14, 20],
    'atr_mult': [1.5, 2.0, 3.0],
    'rsi_period': [2, 5, 14],
    'rsi_thresh': [10, 20, 30],
    'sma_short': [5, 10],
    'sma_long': [100, 200],
}


# ============================================================
# パターン E: 価格 Z-Score (極端なショック吸収)
# ============================================================
def strategy_e_prep(ticker_dfs, z_period, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['Z_SCORE'] = calc_zscore(d['Close'], z_period)
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_e_signal(row, params):
    # Z-Scoreが極端にマイナス && 長期トレンドは上
    return (row['Close'] > row['SMA_LONG'] and
            row.get('Z_SCORE', 0) < params['z_thresh'])

def strategy_e_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    # Z-Scoreが平均(0)に戻った or 損切り or 日数
    return (row.get('Z_SCORE', -99) >= 0 or
            days_held >= params['hold_days'] or
            t_open <= sl_price)

STRATEGY_E_GRID = {
    'z_period': [10, 20, 30],
    'z_thresh': [-2.0, -2.5, -3.0],
    'sma_long': [100, 200],
    'stop_loss': [0.05, 0.08, 0.10],
    'hold_days': [5, 10, 15],
}


# ============================================================
# パターン F: 日次リターン Z-Score (パニック売り検知)
# ============================================================
def strategy_f_prep(ticker_dfs, ret_z_period, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        d['RET_ZSCORE'] = calc_daily_return_zscore(d['Close'], ret_z_period)
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_f_signal(row, params):
    # 特定の日の下落幅が過去データから見て極端な異常値（-2.5σなど）
    return (row['Close'] > row['SMA_LONG'] and
            row.get('RET_ZSCORE', 0) < params['ret_z_thresh'])

def strategy_f_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    # パニック売りのリバウンドは非常に早いため、1〜3日で決済
    return (days_held >= params['hold_days'] or
            t_open <= sl_price)

STRATEGY_F_GRID = {
    'ret_z_period': [20, 50, 100],
    'ret_z_thresh': [-2.5, -3.0, -3.5],
    'sma_long': [100, 200],
    'stop_loss': [0.05, 0.08, 0.10],
    'hold_days': [1, 2, 3, 5],
}


# ============================================================
# パターン G: ボリンジャー %B + ヒストリカル・ボラティリティ (HV)
# ============================================================
def strategy_g_prep(ticker_dfs, bb_period, bb_std, hv_period, sma_long):
    result = {}
    for ticker, df in ticker_dfs.items():
        if len(df) < sma_long:
            continue
        d = df.copy()
        upper, mid, lower = calc_bollinger(d['Close'], bb_period, bb_std)
        # %B: 0以下なら下限割れ
        d['BB_PCT_B'] = (d['Close'] - lower) / (upper - lower)
        d['BB_MID'] = mid
        d['HV'] = calc_historical_volatility(d['Close'], hv_period)
        d['SMA_LONG'] = d['Close'].rolling(sma_long).mean()
        result[ticker] = d
    return result

def strategy_g_signal(row, params):
    # HVが低い(相場が穏やか)のに、突然BB下限を割った(%B<0)
    return (row['Close'] > row['SMA_LONG'] and
            row.get('HV', 999) < params['hv_thresh'] and
            row.get('BB_PCT_B', 1.0) < 0.0)

def strategy_g_exit(row, pos, days_held, t_open, params):
    sl_price = pos['entry_price'] * (1 - params['stop_loss'])
    # バンド中心線(BB_MID)へ回帰 or 損切り
    return (row['Close'] >= row.get('BB_MID', row['Close']) or
            days_held >= params['hold_days'] or
            t_open <= sl_price)

STRATEGY_G_GRID = {
    'bb_period': [20, 50],
    'bb_std': [2.0, 2.5],
    'hv_period': [20, 50],
    'hv_thresh': [0.2, 0.3, 0.4], # HVが20%, 30%, 40%未満
    'sma_long': [100, 200],
    'stop_loss': [0.05, 0.08, 0.10],
    'hold_days': [5, 10, 15],
}


# ============================================================
# メイン：全パターン探索
# ============================================================
def run_pattern_a(ticker_dfs, dates):
    grid = STRATEGY_A_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern A] RSI + SMA: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['rsi_period'], p['sma_short'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_a_prep(ticker_dfs, p['rsi_period'], p['sma_short'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_a_signal, strategy_a_exit, p)
        res['params'] = p
        res['strategy'] = 'A: RSI+SMA'
        results.append(res)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    return results


def run_pattern_b(ticker_dfs, dates):
    grid = STRATEGY_B_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern B] MACD: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['macd_fast'], p['macd_slow'], p['macd_signal'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_b_prep(ticker_dfs, p['macd_fast'], p['macd_slow'], p['macd_signal'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_b_signal, strategy_b_exit, p)
        res['params'] = p
        res['strategy'] = 'B: MACD'
        results.append(res)

    return results


def run_pattern_c(ticker_dfs, dates):
    grid = STRATEGY_C_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern C] Bollinger+RSI: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['bb_period'], p['bb_std'], p['rsi_period'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_c_prep(ticker_dfs, p['bb_period'], p['bb_std'], p['rsi_period'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_c_signal, strategy_c_exit, p)
        res['params'] = p
        res['strategy'] = 'C: BB+RSI'
        results.append(res)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    return results


def run_pattern_d(ticker_dfs, dates):
    grid = STRATEGY_D_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern D] ATR+RSI: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['atr_period'], p['rsi_period'], p['sma_short'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_d_prep(ticker_dfs, p['atr_period'], p['rsi_period'], p['sma_short'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_d_signal, strategy_d_exit, p)
        res['params'] = p
        res['strategy'] = 'D: ATR+RSI'
        results.append(res)

    return results


def run_pattern_e(ticker_dfs, dates):
    grid = STRATEGY_E_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern E] Price Z-Score: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['z_period'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_e_prep(ticker_dfs, p['z_period'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_e_signal, strategy_e_exit, p)
        res['params'] = p
        res['strategy'] = 'E: Price Z-Score'
        results.append(res)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    return results


def run_pattern_f(ticker_dfs, dates):
    grid = STRATEGY_F_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern F] Return Z-Score: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['ret_z_period'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_f_prep(ticker_dfs, p['ret_z_period'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_f_signal, strategy_f_exit, p)
        res['params'] = p
        res['strategy'] = 'F: Return Z-Score'
        results.append(res)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    return results


def run_pattern_g(ticker_dfs, dates):
    grid = STRATEGY_G_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n[Pattern G] BB %B + HV: {len(combos)} combinations")

    results = []
    cache = {}
    for i, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        cache_key = (p['bb_period'], p['bb_std'], p['hv_period'], p['sma_long'])
        if cache_key not in cache:
            cache[cache_key] = strategy_g_prep(ticker_dfs, p['bb_period'], p['bb_std'], p['hv_period'], p['sma_long'])
        indicators = cache[cache_key]

        res = run_backtest(indicators, dates, strategy_g_signal, strategy_g_exit, p)
        res['params'] = p
        res['strategy'] = 'G: BB %B + HV'
        results.append(res)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    return results


def main():
    t0 = time.time()
    ticker_dfs, dates = fetch_data()

    all_results = []
    all_results.extend(run_pattern_a(ticker_dfs, dates))
    all_results.extend(run_pattern_b(ticker_dfs, dates))
    all_results.extend(run_pattern_c(ticker_dfs, dates))
    all_results.extend(run_pattern_d(ticker_dfs, dates))
    all_results.extend(run_pattern_e(ticker_dfs, dates))
    all_results.extend(run_pattern_f(ticker_dfs, dates))
    all_results.extend(run_pattern_g(ticker_dfs, dates))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total: {len(all_results)} combinations tested in {elapsed:.1f}s")
    print(f"{'='*70}")

    # 各パターンの最良結果
    strategies = [
        'A: RSI+SMA', 'B: MACD', 'C: BB+RSI', 'D: ATR+RSI',
        'E: Price Z-Score', 'F: Return Z-Score', 'G: BB %B + HV'
    ]
    print("\n=== Best per Strategy ===")
    best_per_strategy = []
    for s in strategies:
        s_results = [r for r in all_results if r['strategy'] == s]
        if not s_results:
            continue
        best = max(s_results, key=lambda x: x['return'])
        best_per_strategy.append(best)
        print(f"\n[{s}]")
        print(f"  Return: {best['return']:.2f}%  |  Trades: {best['trades']}  |  Win Rate: {best['win_rate']:.1f}%  |  Avg Return/Trade: {best['avg_return']:.2f}%")
        print(f"  Params: {best['params']}")

    # 全体ランキング Top 10
    all_results.sort(key=lambda x: x['return'], reverse=True)
    print(f"\n{'='*70}")
    print("=== Overall Top 10 ===")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Strategy':<15} {'Return':>8} {'Trades':>7} {'WinRate':>8} {'AvgRet':>8}  Params")
    print("-" * 100)
    for i, r in enumerate(all_results[:10], 1):
        p_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
        print(f"{i:<5} {r['strategy']:<15} {r['return']:>7.2f}% {r['trades']:>7} {r['win_rate']:>7.1f}% {r['avg_return']:>7.2f}%  {p_str}")

    # Worst 5 for comparison
    print(f"\n=== Bottom 5 ===")
    for i, r in enumerate(all_results[-5:], 1):
        p_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
        print(f"{i:<5} {r['strategy']:<15} {r['return']:>7.2f}% {r['trades']:>7} {r['win_rate']:>7.1f}% {r['avg_return']:>7.2f}%  {p_str}")

    # Best overall recommendation
    print(f"\n{'='*70}")
    print("🏆 RECOMMENDED BEST PARAMETERS:")
    print(f"{'='*70}")
    best = all_results[0]
    print(f"  Strategy  : {best['strategy']}")
    print(f"  Return    : {best['return']:.2f}%")
    print(f"  Win Rate  : {best['win_rate']:.1f}%")
    print(f"  Trades    : {best['trades']}")
    print(f"  Parameters:")
    for k, v in best['params'].items():
        print(f"    {k}: {v}")


if __name__ == '__main__':
    main()
