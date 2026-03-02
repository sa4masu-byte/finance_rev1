import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from src import config
from datetime import datetime

@dataclass
class Signal:
    ticker: str
    date: str
    close_price: float
    rsi2: float
    sma5: float
    sma200: float
    score: float
    reason: str
    special_alert: str = ""

class SwingAnalyzer:
    def __init__(self, tickers: List[str] = config.TARGET_TICKERS):
        self.tickers = list(dict.fromkeys(tickers))  # 重複除去（順序保持）
        
    def _calc_rsi(self, series: pd.Series, period: int = 2) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calc_zscore(self, series: pd.Series, period: int) -> pd.Series:
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return (series - sma) / std
        
    def _calc_daily_return_zscore(self, series: pd.Series, period: int) -> pd.Series:
        daily_returns = series.pct_change()
        mean_return = daily_returns.rolling(period).mean()
        std_return = daily_returns.rolling(period).std()
        return (daily_returns - mean_return) / std_return

    def _calc_historical_volatility(self, series: pd.Series, period: int) -> pd.Series:
        daily_returns = series.pct_change()
        return daily_returns.rolling(period).std() * np.sqrt(252)

    def _calc_bollinger(self, series: pd.Series, period: int, std_dev: float):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    def get_market_data(self) -> pd.DataFrame:
        """yfinanceから半年分のデータを一括取得"""
        print("Fetching market data...")
        data = yf.download(
            self.tickers, 
            period="1y", 
            group_by='ticker', 
            progress=False,
            auto_adjust=False
        )
        return data

    def analyze(self) -> List[Signal]:
        """全対象銘柄を分析し、買いシグナルが出ているものを抽出"""
        data = self.get_market_data()
        signals = []
        
        for ticker in self.tickers:
            try:
                # MultiIndex対応 (yfinance.downloadの戻り値構造)
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.levels[0]:
                        continue
                    df = data[ticker].copy()
                else:
                    if len(self.tickers) == 1:
                        df = data.copy()
                    else:
                        continue
                        
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < config.P_SMA_LONG:
                    continue
                    
                close = df['Close']
                df['SMA5'] = close.rolling(config.P_SMA_SHORT).mean()
                df['SMA200'] = close.rolling(config.P_SMA_LONG).mean()
                df['RSI2'] = self._calc_rsi(close, config.P_RSI)
                
                # 追加：E,F,G 用の指標
                df['Z_SCORE'] = self._calc_zscore(close, config.E_Z_PERIOD)
                df['RET_ZSCORE'] = self._calc_daily_return_zscore(close, config.F_RET_Z_PERIOD)
                upper, mid, lower = self._calc_bollinger(close, config.G_BB_PERIOD, config.G_BB_STD)
                df['BB_PCT_B'] = (close - lower) / (upper - lower)
                df['HV'] = self._calc_historical_volatility(close, config.G_HV_PERIOD)
                
                # 最新日のデータを取得
                latest = df.iloc[-1]
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                
                c_price = latest['Close']
                sma200 = latest['SMA200']
                sma5 = latest['SMA5']
                rsi2 = latest['RSI2']
                z_score = latest['Z_SCORE']
                ret_zscore = latest['RET_ZSCORE']
                bb_pct_b = latest['BB_PCT_B']
                hv = latest['HV']
                
                # シグナル判定条件（Connors RSI + Trend filter）
                is_strategy_a = (c_price > sma200 and c_price < sma5 and rsi2 < config.RSI_THRESHOLD)
                
                # E,F,G 特別パニック検知
                is_strategy_e = (c_price > sma200 and z_score < config.E_Z_THRESH)
                is_strategy_f = (c_price > sma200 and ret_zscore < config.F_RET_Z_THRESH)
                is_strategy_g = (c_price > sma200 and hv < config.G_HV_THRESH and bb_pct_b < 0.0)

                if is_strategy_a or is_strategy_e or is_strategy_f or is_strategy_g:
                    
                    # 総合スコア(100点満点)の算出
                    # RSIが低い（売られすぎ）ほど高得点: Max 50点
                    score_rsi = max(0, (config.RSI_THRESHOLD - rsi2) * (50 / config.RSI_THRESHOLD)) if not pd.isna(rsi2) else 0
                    
                    # トレンドSMAとの乖離（トレンドの強さ）: Max 50点
                    trend_diff_pct = ((c_price - sma200) / sma200) * 100
                    score_trend = min(50, max(0, trend_diff_pct * (50 / 20)))
                    
                    total_score = round(score_rsi + score_trend, 1)
                    
                    special_alert = ""
                    reason_lines = []
                    
                    if is_strategy_e:
                        special_alert += f"🚨【勝率100%クラス: Z-Scoreショック】価格Z-Scoreが {z_score:.1f}σ (閾値: {config.E_Z_THRESH}σ) と極端な売られすぎです！高い確率で反発が期待できます。\n"
                        total_score += 1000  # ランキング1位に固定
                    if is_strategy_f:
                        special_alert += f"🚨【高勝率パニック検知】日次リターンのZ-Scoreが {ret_zscore:.1f}σ (閾値: {config.F_RET_Z_THRESH}σ) を記録。強烈な狼狽売りの可能性大！\n"
                        total_score += 1000
                    if is_strategy_g:
                        special_alert += f"🚨【勝率100%クラス: サイレント・ブレイク】低ボラティリティ相場 (HV: {hv*100:.1f}%) で突如BB下限を割る異常が発生しました。\n"
                        total_score += 1000

                    if is_strategy_a:
                        reason_lines.append(f"短期的な売られすぎ（RSI({config.P_RSI}): {rsi2:.1f}）による反発期待に加え、中長期（{config.P_SMA_LONG}日線比 +{trend_diff_pct:.1f}%）の上昇トレンドの押し目。")
                    else:
                        reason_lines.append(f"中長期（{config.P_SMA_LONG}日線比 +{trend_diff_pct:.1f}%）の上昇トレンド維持。RSIは通常水準ですが、上記の特別なパニック指標が点灯しています。")
                        
                    reason_text = " ".join(reason_lines)
                    
                    sig = Signal(
                        ticker=ticker,
                        date=latest_date,
                        close_price=c_price,
                        rsi2=rsi2,
                        sma5=sma5,
                        sma200=sma200,
                        score=total_score,
                        reason=reason_text,
                        special_alert=special_alert.strip()
                    )
                    signals.append(sig)
                    
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
                
        # 総合スコアが高い順（より優位性が高い）にソート → 上位MAX_POSITIONSに絞る
        signals.sort(key=lambda x: x.score, reverse=True)
        return signals[:config.MAX_POSITIONS]
