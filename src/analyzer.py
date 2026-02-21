import yfinance as yf
import pandas as pd
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

class SwingAnalyzer:
    def __init__(self, tickers: List[str] = config.TARGET_TICKERS):
        self.tickers = tickers
        
    def _calc_rsi(self, series: pd.Series, period: int = 2) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_market_data(self) -> pd.DataFrame:
        """yfinanceから半年分のデータを一括取得"""
        print("Fetching market data...")
        data = yf.download(
            self.tickers, 
            period="6mo", 
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
                
                # 最新日のデータを取得
                latest = df.iloc[-1]
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                
                c_price = latest['Close']
                sma200 = latest['SMA200']
                sma5 = latest['SMA5']
                rsi2 = latest['RSI2']
                
                # シグナル判定条件（Connors RSI + Trend filter）
                # 1. 200日線より上（全体は上昇トレンド）
                # 2. 5日線より下（手前は押し目・調整中）
                # 3. RSI(2)が10未満（極端な売られすぎ）
                if c_price > sma200 and c_price < sma5 and rsi2 < config.RSI_THRESHOLD:
                    sig = Signal(
                        ticker=ticker,
                        date=latest_date,
                        close_price=c_price,
                        rsi2=rsi2,
                        sma5=sma5,
                        sma200=sma200
                    )
                    signals.append(sig)
                    
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
                
        # RSIが低い順（より優位性が高い）にソート
        signals.sort(key=lambda x: x.rsi2)
        return signals
