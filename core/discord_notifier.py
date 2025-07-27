import requests

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send(self, message):
        payload = {"content": message}
        try:
            requests.post(self.webhook_url, json=payload)
        except Exception as e:
            print(f"Discord notification failed: {e}")
