import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import itertools
from src import config
from concurrent.futures import ProcessPoolExecutor

# 対象銘柄
TICKERS = config.TARGET_TICKERS

# テストパラメータの組み合わせ （RSI閾値, 短期SMA, 損切り率）
PARAMS_OPTIONS = {
    'rsi': [5, 10, 15],
    'sma_short': [3, 5, 7],
    'stop_loss_pct': [0.03, 0.05, 0.08]
}

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_prep_data():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    print(f"Fetching data from {start_date} to {end_date} for {len(TICKERS)} tickers...")
    data = yf.download(TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False)
    
    eval_start_date = pd.to_datetime(end_date - datetime.timedelta(days=90)).tz_localize('Asia/Tokyo')
    try:
        data.index = data.index.tz_convert('Asia/Tokyo')
    except Exception:
        data.index = data.index.tz_localize('Asia/Tokyo')
        
    dates = data.index.unique().sort_values()
    dates = [d for d in dates if d >= eval_start_date]
    
    return data, dates, eval_start_date

def run_simulation(data, dates, rsi_thresh, sma_short, sl_pct):
    # indicators の計算
    indicators = {}
    for ticker in TICKERS:
        try:
            df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) < config.P_SMA_LONG:
                continue
            df['SMA_SHORT'] = df['Close'].rolling(sma_short).mean()
            df['SMA_LONG'] = df['Close'].rolling(config.P_SMA_LONG).mean()
            df['RSI'] = calc_rsi(df['Close'], config.P_RSI)
            indicators[ticker] = df
        except Exception:
            pass

    initial_capital = 1000000
    capital = initial_capital
    positions = {}
    trade_history = []
    
    for today in dates:
        # 1. 売却処理
        yesterday_idx = dates.index(today) - 1
        if yesterday_idx >= 0:
            yesterday = dates[yesterday_idx]
            sold_tickers = []
            for ticker, pos in positions.items():
                df = indicators[ticker]
                if yesterday in df.index:
                    y_row = df.loc[yesterday]
                    days_held = (today - pos['entry_date']).days
                    if today in df.index:
                        t_open = df.loc[today, 'Open']
                    else:
                        continue
                        
                    stop_loss_price = pos['entry_price'] * (1 - sl_pct)
                    
                    if y_row['Close'] > y_row['SMA_SHORT'] or days_held >= 10 or t_open <= stop_loss_price:
                        sell_price = t_open
                        profit = (sell_price - pos['entry_price']) * pos['shares']
                        capital += sell_price * pos['shares']
                        trade_history.append({
                            'profit': profit,
                            'return_pct': (sell_price / pos['entry_price'] - 1) * 100
                        })
                        sold_tickers.append(ticker)
            for t in sold_tickers:
                del positions[t]

        # 2. 購入処理
        available_cash = capital
        max_position_size = initial_capital * 0.20
        
        buy_candidates = []
        for ticker, df in indicators.items():
            if ticker in positions:
                continue
            yesterday_idx = dates.index(today) - 1
            if yesterday_idx < 0: continue
            yesterday = dates[yesterday_idx]
            
            if yesterday in df.index and today in df.index:
                y_row = df.loc[yesterday]
                if (y_row['Close'] > y_row['SMA_LONG'] and 
                    y_row['Close'] < y_row['SMA_SHORT'] and 
                    y_row['RSI'] < rsi_thresh):
                    
                    t_open = df.loc[today, 'Open']
                    if pd.isna(t_open): continue
                    
                    # Score calculation
                    score_rsi = max(0, (rsi_thresh - y_row['RSI']) * (50 / rsi_thresh))
                    trend_diff_pct = ((y_row['Close'] - y_row['SMA_LONG']) / y_row['SMA_LONG']) * 100
                    score_trend = min(50, max(0, trend_diff_pct * (50 / 20)))
                    total_score = round(score_rsi + score_trend, 1)
                    
                    buy_candidates.append((ticker, t_open, total_score))
                    
        buy_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for ticker, t_open, score in buy_candidates:
            if len(positions) >= config.MAX_POSITIONS:
                break
            
            max_shares_by_rule = int(max_position_size // (t_open * 100)) * 100
            max_shares_by_cash = int(available_cash // (t_open * 100)) * 100
            shares = min(max_shares_by_rule, max_shares_by_cash)
            
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
    last_date = dates[-1]
    for ticker, pos in list(positions.items()):
        df = indicators[ticker]
        if last_date in df.index:
            t_close = df.loc[last_date, 'Close']
            profit = (t_close - pos['entry_price']) * pos['shares']
            capital += t_close * pos['shares']
            trade_history.append({
                'profit': profit,
                'return_pct': (t_close / pos['entry_price'] - 1) * 100
            })
            del positions[ticker]
            
    total_return_pct = ((capital / initial_capital) - 1) * 100
    win_rate = 0 if len(trade_history) == 0 else (len([t for t in trade_history if t['profit'] > 0]) / len(trade_history) * 100)
    
    return {
        'rsi': rsi_thresh,
        'sma': sma_short,
        'sl': sl_pct,
        'return': total_return_pct,
        'trades': len(trade_history),
        'win_rate': win_rate
    }

def main():
    data, dates, _ = fetch_and_prep_data()
    
    combinations = list(itertools.product(
        PARAMS_OPTIONS['rsi'], 
        PARAMS_OPTIONS['sma_short'], 
        PARAMS_OPTIONS['stop_loss_pct']
    ))
    
    print(f"Starting Grid Search for {len(combinations)} combinations...")
    
    results = []
    # 直列実行（メモリ制限など考慮）
    for i, (rsi, sma, sl) in enumerate(combinations, 1):
        # logging
        if i % 5 == 0:
            print(f"Running {i}/{len(combinations)}...")
            
        res = run_simulation(data, dates, rsi, sma, sl)
        results.append(res)
        
    # Sort by total return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print("\n=== Top 5 Parameter Sets ===")
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. RSI<{res['rsi']:<2} | SMA{res['sma']} | SL -{int(res['sl']*100)}% => Return: {res['return']:.2f}% | Win Rate: {res['win_rate']:.2f}% | Trades: {res['trades']}")

if __name__ == '__main__':
    main()
