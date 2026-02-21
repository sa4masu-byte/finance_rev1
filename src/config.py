import os
from typing import List

# 対象銘柄（流動性・業績の安定したTOPIX Core30相当レベルの大型優良株 30銘柄）
# 本番運用ではさらに増やすことも可能ですが、ここではバックテストで優位性が確認された30銘柄をデフォルトとします。
TARGET_TICKERS: List[str] = [
    "7203.T", "8306.T", "6861.T", "6758.T", "9984.T", 
    "9432.T", "8058.T", "6920.T", "7974.T", "4063.T",
    "8316.T", "8031.T", "6501.T", "8001.T", "9433.T",
    "9983.T", "6098.T", "4502.T", "8766.T", "6902.T",
    "7011.T", "6702.T", "8411.T", "4568.T", "4519.T",
    "6502.T", "8002.T", "6146.T", "4543.T", "6954.T"
]

# LINE Messaging API トークン
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.getenv("LINE_USER_ID", "")

# 戦略パラメータ
P_SMA_LONG = 200    # 長期トレンド判定用
P_SMA_SHORT = 5     # 短期トレンド判定用
P_RSI = 2           # 短期オシレータ用
RSI_THRESHOLD = 10  # 超過売られすぎ水準

# 最大保有銘柄数（資金分散）
MAX_POSITIONS = 5
