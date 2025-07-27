import requests

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.token = bot_token
        self.chat_id = chat_id

    def send(self, message):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Telegram notification failed: {e}")
