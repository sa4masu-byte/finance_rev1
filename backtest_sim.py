import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# 対象銘柄（流動性の高い大型株メイン TOPIX Core30 相当）
TICKERS = [
    "7203.T", "8306.T", "6861.T", "6758.T", "9984.T", 
    "9432.T", "8058.T", "6920.T", "7974.T", "4063.T",
    "8316.T", "8031.T", "6501.T", "8001.T", "9433.T",
    "9983.T", "6098.T", "4502.T", "8766.T", "6902.T",
    "7011.T", "6702.T", "8411.T", "4568.T", "4519.T",
    "6502.T", "8002.T", "6146.T", "4543.T", "6954.T"
]

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_backtest():
    end_date = datetime.date(2026, 2, 21)
    # 検証期間は少し長めに半年とって、直近3ヶ月を評価区間にする
    start_date = end_date - datetime.timedelta(days=365)
    
    print(f"Fetching data from {start_date} to {end_date}...")
    data = yf.download(TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False)
    
    # 評価開始日 (約3ヶ月前)
    eval_start_date = pd.to_datetime(end_date - datetime.timedelta(days=90)).tz_localize('Asia/Tokyo')
    try:
        data.index = data.index.tz_convert('Asia/Tokyo')
    except Exception:
        data.index = data.index.tz_localize('Asia/Tokyo')
        
    initial_capital = 1000000
    capital = initial_capital
    positions = {} # ticker: {shares, entry_price, entry_date}
    trade_history = []
    
    # 日付のリストを取得
    dates = data.index.unique().sort_values()
    dates = [d for d in dates if d >= eval_start_date]
        
    # 指標の事前計算
    indicators = {}
    for ticker in TICKERS:
        try:
            df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) < 50:
                continue
            df['SMA5'] = df['Close'].rolling(5).mean()
            df['SMA25'] = df['Close'].rolling(25).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['RSI2'] = calc_rsi(df['Close'], 2)
            indicators[ticker] = df
        except Exception as e:
            pass

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
                        
                    stop_loss_price = pos['entry_price'] * 0.95
                    
                    if y_row['Close'] > y_row['SMA5'] or days_held >= 10 or t_open <= stop_loss_price:
                        sell_price = t_open
                        profit = (sell_price - pos['entry_price']) * pos['shares']
                        capital += sell_price * pos['shares']
                        trade_history.append({
                            'ticker': ticker,
                            'entry_date': pos['entry_date'],
                            'exit_date': today,
                            'entry_price': pos['entry_price'],
                            'exit_price': sell_price,
                            'shares': pos['shares'],
                            'profit': profit,
                            'return_pct': (sell_price / pos['entry_price'] - 1) * 100
                        })
                        sold_tickers.append(ticker)
            for t in sold_tickers:
                del positions[t]

        # 2. 購入処理
        available_cash = capital
        max_position_size = initial_capital * 0.20 # 1銘柄最大20万円
        
        buy_candidates = []
        for ticker, df in indicators.items():
            if ticker in positions:
                continue
            yesterday_idx = dates.index(today) - 1
            if yesterday_idx < 0: continue
            yesterday = dates[yesterday_idx]
            
            if yesterday in df.index and today in df.index:
                y_row = df.loc[yesterday]
                if (y_row['Close'] > y_row['SMA200'] and 
                    y_row['Close'] < y_row['SMA5'] and 
                    y_row['RSI2'] < 10):
                    t_open = df.loc[today, 'Open']
                    if pd.isna(t_open): continue
                    score = y_row['RSI2']
                    buy_candidates.append((ticker, t_open, score))
                    
        buy_candidates.sort(key=lambda x: x[2])
        
        for ticker, t_open, score in buy_candidates:
            if len(positions) >= 5:
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
                
    # 最後の日に全決済
    last_date = dates[-1]
    for ticker, pos in list(positions.items()):
        df = indicators[ticker]
        if last_date in df.index:
            t_close = df.loc[last_date, 'Close']
            profit = (t_close - pos['entry_price']) * pos['shares']
            capital += t_close * pos['shares']
            trade_history.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': last_date,
                'entry_price': pos['entry_price'],
                'exit_price': t_close,
                'shares': pos['shares'],
                'profit': profit,
                'return_pct': (t_close / pos['entry_price'] - 1) * 100
            })
            del positions[ticker]
            
    print("=== Backtest Results (Last 3 Months) ===")
    print(f"Initial Capital: ¥{initial_capital:,}")
    print(f"Final Capital  : ¥{int(capital):,}")
    print(f"Total Return   : {((capital / initial_capital) - 1) * 100:.2f}%")
    
    if trade_history:
        wins = [t for t in trade_history if t['profit'] > 0]
        losses = [t for t in trade_history if t['profit'] <= 0]
        win_rate = len(wins) / len(trade_history) * 100
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
        
        print(f"Total Trades   : {len(trade_history)}")
        print(f"Win Rate       : {win_rate:.2f}%")
        print(f"Avg Win Return : {avg_win:.2f}%")
        print(f"Avg Loss Return: {avg_loss:.2f}%")
    else:
        print("No trades executed.")
        
if __name__ == '__main__':
    run_backtest()
