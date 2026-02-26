"""
バックテスト詳細シミュレーション
- 初期資産100万円、信用取引（手元資金の3倍まで）
- 最適化済みパラメータ使用
- 全トレード詳細を出力
"""
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from src import config

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest():
    end_date = datetime.date(2026, 2, 26)
    start_date = end_date - datetime.timedelta(days=730)  # 2年分取得（SMA計算用の余裕）

    tickers = list(dict.fromkeys(config.TARGET_TICKERS))
    print(f"Fetching data from {start_date} to {end_date} for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=False)

    eval_start_date = pd.to_datetime(end_date - datetime.timedelta(days=365)).tz_localize('Asia/Tokyo')
    try:
        data.index = data.index.tz_convert('Asia/Tokyo')
    except Exception:
        data.index = data.index.tz_localize('Asia/Tokyo')

    # === 信用取引設定 ===
    LEVERAGE = 3           # 手元資金の3倍まで取引可能
    initial_capital = 1_000_000
    capital = initial_capital  # 現金残高
    buying_power = initial_capital * LEVERAGE  # 総取引可能額
    max_position_size = buying_power * 0.20  # 1銘柄最大（総枠の20%）

    positions = {}  # ticker: {shares, entry_price, entry_date}
    trade_history = []

    dates = sorted(data.index.unique())
    dates = [d for d in dates if d >= eval_start_date]

    # 指標計算
    indicators = {}
    for ticker in tickers:
        try:
            df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) < config.P_SMA_LONG:
                continue
            df['SMA_SHORT'] = df['Close'].rolling(config.P_SMA_SHORT).mean()
            df['SMA_LONG'] = df['Close'].rolling(config.P_SMA_LONG).mean()
            df['RSI'] = calc_rsi(df['Close'], config.P_RSI)
            indicators[ticker] = df
        except Exception:
            pass

    print(f"Usable tickers: {len(indicators)}")
    print(f"Eval period: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ¥{initial_capital:,}")
    print(f"Leverage: {LEVERAGE}x → Buying Power: ¥{int(buying_power):,}")
    print()

    for idx, today in enumerate(dates):
        # === 売却処理 ===
        if idx > 0:
            yesterday = dates[idx - 1]
            sold = []
            for ticker, pos in positions.items():
                df = indicators[ticker]
                if yesterday not in df.index or today not in df.index:
                    continue
                y_row = df.loc[yesterday]
                t_open = df.loc[today, 'Open']
                if pd.isna(t_open):
                    continue

                days_held = (today - pos['entry_date']).days
                stop_loss_price = pos['entry_price'] * (1 - config.STOP_LOSS_PCT)

                sell_reason = None
                if y_row['Close'] > y_row['SMA_SHORT']:
                    sell_reason = "SMA回帰"
                elif days_held >= config.HOLD_DAYS_MAX:
                    sell_reason = f"保有上限{config.HOLD_DAYS_MAX}日"
                elif t_open <= stop_loss_price:
                    sell_reason = f"損切り({config.STOP_LOSS_PCT*100:.0f}%)"

                if sell_reason:
                    sell_price = t_open
                    profit = (sell_price - pos['entry_price']) * pos['shares']
                    return_pct = (sell_price / pos['entry_price'] - 1) * 100
                    capital += sell_price * pos['shares']

                    trade_history.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': today,
                        'entry_price': pos['entry_price'],
                        'exit_price': sell_price,
                        'shares': pos['shares'],
                        'cost': pos['entry_price'] * pos['shares'],
                        'profit': profit,
                        'return_pct': return_pct,
                        'days_held': days_held,
                        'exit_reason': sell_reason,
                    })
                    sold.append(ticker)
            for t in sold:
                del positions[t]

        # === 購入処理 ===
        if idx == 0:
            continue
        yesterday = dates[idx - 1]

        # 現在のポジション評価額
        total_position_value = sum(
            indicators[t].loc[today, 'Close'] * p['shares']
            for t, p in positions.items()
            if today in indicators[t].index
        ) if positions else 0

        # 信用取引枠: 現金 + 含み益/損 を元にした枠
        equity = capital + total_position_value - sum(p['entry_price'] * p['shares'] for p in positions.values())
        available_buying_power = equity * LEVERAGE - total_position_value
        available_cash_for_buy = min(capital, available_buying_power)

        candidates = []
        for ticker, df in indicators.items():
            if ticker in positions:
                continue
            if yesterday not in df.index or today not in df.index:
                continue
            y_row = df.loc[yesterday]
            if (y_row['Close'] > y_row['SMA_LONG'] and
                y_row['Close'] < y_row['SMA_SHORT'] and
                y_row['RSI'] < config.RSI_THRESHOLD):

                t_open = df.loc[today, 'Open']
                if pd.isna(t_open):
                    continue

                rsi_val = y_row['RSI']
                score_rsi = max(0, (config.RSI_THRESHOLD - rsi_val) * (50 / config.RSI_THRESHOLD))
                trend_diff = ((y_row['Close'] - y_row['SMA_LONG']) / y_row['SMA_LONG']) * 100
                score_trend = min(50, max(0, trend_diff * (50 / 20)))
                score = round(score_rsi + score_trend, 1)

                candidates.append((ticker, t_open, score, rsi_val))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for ticker, t_open, score, rsi_val in candidates:
            if len(positions) >= config.MAX_POSITIONS:
                break

            max_s_rule = int(max_position_size // (t_open * 100)) * 100
            max_s_cash = int(available_cash_for_buy // (t_open * 100)) * 100
            shares = min(max_s_rule, max_s_cash)

            if shares >= 100:
                cost = shares * t_open
                capital -= cost
                available_cash_for_buy -= cost
                positions[ticker] = {
                    'shares': shares,
                    'entry_price': t_open,
                    'entry_date': today,
                    'score': score,
                    'rsi': rsi_val,
                }

    # 最終日に全決済
    last_date = dates[-1]
    for ticker, pos in list(positions.items()):
        df = indicators[ticker]
        if last_date in df.index:
            t_close = df.loc[last_date, 'Close']
            profit = (t_close - pos['entry_price']) * pos['shares']
            return_pct = (t_close / pos['entry_price'] - 1) * 100
            days_held = (last_date - pos['entry_date']).days
            capital += t_close * pos['shares']
            trade_history.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': last_date,
                'entry_price': pos['entry_price'],
                'exit_price': t_close,
                'shares': pos['shares'],
                'cost': pos['entry_price'] * pos['shares'],
                'profit': profit,
                'return_pct': return_pct,
                'days_held': days_held,
                'exit_reason': '最終日決済',
            })

    # ============================================================
    #  結果出力
    # ============================================================
    print("=" * 100)
    print("  バックテスト詳細結果（信用取引 3倍レバレッジ）")
    print("=" * 100)

    # 全トレード一覧
    print(f"\n{'No':>3} {'買日':>10}  {'売日':>10}  {'銘柄':>8}  {'株数':>5}  {'買値':>8}  {'売値':>8}  {'損益':>9}  {'収益率':>7}  {'日数':>3}  売却理由")
    print("-" * 100)

    total_profit = 0
    for i, t in enumerate(trade_history, 1):
        entry_d = t['entry_date'].strftime('%Y-%m-%d')
        exit_d = t['exit_date'].strftime('%Y-%m-%d')
        pnl_mark = "+" if t['profit'] >= 0 else ""
        print(f"{i:>3} {entry_d}  {exit_d}  {t['ticker']:>8}  {t['shares']:>5}  "
              f"¥{t['entry_price']:>7,.0f}  ¥{t['exit_price']:>7,.0f}  "
              f"{pnl_mark}¥{t['profit']:>8,.0f}  {t['return_pct']:>+6.2f}%  "
              f"{t['days_held']:>3}日  {t['exit_reason']}")
        total_profit += t['profit']

    # サマリー
    print()
    print("=" * 100)
    print("  サマリー")
    print("=" * 100)
    final_capital = capital
    wins = [t for t in trade_history if t['profit'] > 0]
    losses = [t for t in trade_history if t['profit'] <= 0]

    print(f"  初期資産         : ¥{initial_capital:>12,}")
    print(f"  最終資産         : ¥{int(final_capital):>12,}")
    print(f"  損益合計         : ¥{int(total_profit):>12,}")
    print(f"  トータルリターン : {((final_capital / initial_capital) - 1) * 100:>+.2f}%")
    print()
    print(f"  取引回数         : {len(trade_history)}回")
    print(f"  勝ちトレード     : {len(wins)}回")
    print(f"  負けトレード     : {len(losses)}回")
    print(f"  勝率             : {len(wins)/len(trade_history)*100:.1f}%" if trade_history else "  勝率: N/A")

    if wins:
        print(f"  平均勝ちリターン : {np.mean([t['return_pct'] for t in wins]):>+.2f}%")
        print(f"  最大勝ち         : ¥{max(t['profit'] for t in wins):>+,.0f}")
    if losses:
        print(f"  平均負けリターン : {np.mean([t['return_pct'] for t in losses]):>+.2f}%")
        print(f"  最大負け         : ¥{min(t['profit'] for t in losses):>+,.0f}")
    if wins and losses:
        avg_win = np.mean([t['profit'] for t in wins])
        avg_loss = abs(np.mean([t['profit'] for t in losses]))
        pf = (avg_win * len(wins)) / (avg_loss * len(losses)) if avg_loss > 0 else float('inf')
        print(f"  プロフィットファクター: {pf:.2f}")

    print(f"  平均保有日数     : {np.mean([t['days_held'] for t in trade_history]):.1f}日" if trade_history else "")
    print()

    # 月次リターン
    if trade_history:
        print("  【月次損益】")
        monthly = {}
        for t in trade_history:
            m = t['exit_date'].strftime('%Y-%m')
            monthly.setdefault(m, 0)
            monthly[m] += t['profit']
        for m in sorted(monthly):
            bar = "█" * max(1, int(abs(monthly[m]) / 5000))
            sign = "+" if monthly[m] >= 0 else ""
            color = "🟢" if monthly[m] >= 0 else "🔴"
            print(f"    {m}: {color} {sign}¥{monthly[m]:>10,.0f}  {bar}")
    print("=" * 100)


if __name__ == '__main__':
    run_backtest()
