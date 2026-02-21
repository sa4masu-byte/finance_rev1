import requests
from src import config

class LineNotifier:
    def __init__(self, token: str = config.LINE_CHANNEL_ACCESS_TOKEN, user_id: str = config.LINE_USER_ID):
        self.token = token
        self.user_id = user_id
        self.api_url = "https://api.line.me/v2/bot/message/push"
        
    def _send(self, message: str):
        if not self.token or not self.user_id:
            print("LINE_CHANNEL_ACCESS_TOKEN or LINE_USER_ID is not set. Skipping notification.")
            print(f"--- MSG ---\n{message}\n-----------")
            return
            
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "to": self.user_id,
            "messages": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            print("Successfully sent message to LINE Messaging API.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send message: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")

    def notify_signals(self, signals: list):
        if not signals:
            msg = "【J-Swing Insight】\n本日の推奨銘柄はありません。\n(条件を満たす銘柄が0件でした)"
            self._send(msg)
            return
            
        msg = f"【J-Swing Insight 推奨銘柄】\n本日（{signals[0].date}）のシグナル点灯銘柄は以下の通りです:\n\n"
        
        # 上位N件に絞る（config.MAX_POSITIONS）
        top_signals = signals[:config.MAX_POSITIONS]
        
        for i, s in enumerate(top_signals, 1):
            sl_price = int(s.close_price * 0.95) # 推奨損切り（5%下落）
            msg += f"■ {i}. {s.ticker} ⭐{s.score}点\n"
            msg += f"  終値: {int(s.close_price):,}円\n"
            msg += f"  目安損切り: {sl_price:,}円\n"
            msg += f"  理由: {s.reason}\n\n"
            
        if len(signals) > config.MAX_POSITIONS:
            msg += f"※他 {len(signals) - config.MAX_POSITIONS} 件のシグナル点灯あり。\n"
            
        msg += "--- \n※投資は自己責任でお願いします。"
        self._send(msg)
