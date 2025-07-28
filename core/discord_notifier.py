# utils/discord_notifier.py

import requests

class DiscordNotifier:
    """
    Institutional Discord/alert notifier.
    - Supports plain, embed, and formatted messages
    - Defensive, plug-and-play for any bot/alert use
    """
    def __init__(self, webhook_url, username=None, avatar_url=None):
        self.webhook_url = webhook_url
        self.username = username or "Bot"
        self.avatar_url = avatar_url

    def send(self, message, embed=None, codeblock=False):
        """
        Sends a message (plain or embed) to Discord webhook.
        """
        payload = {
            "content": f"```{message}```" if codeblock else message,
            "username": self.username,
        }
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url
        if embed:
            payload["embeds"] = [embed]

        try:
            resp = requests.post(self.webhook_url, json=payload)
            if not resp.ok:
                print(f"Discord notification failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Discord notification failed: {e}")

    def send_trade_alert(self, trade):
        """
        Specialized sender for trade dicts from your bot.
        """
        side = trade.get("side", "N/A")
        pair = trade.get("pair", "UNKNOWN")
        pnl = trade.get("pnl", "N/A")
        conf = trade.get("confidence", "unknown")
        tag = trade.get("pattern", {}).get("type", "")
        reasons = "\n".join(f"- {r}" for r in trade.get("reasons", []))
        msg = (
            f"**Trade Executed:** {pair} {side}\n"
            f"PnL: `{pnl}` | Confidence: `{conf}`\n"
            f"Pattern: `{tag}`\n"
            f"Reasons:\n{reasons}"
        )
        self.send(msg)

    def send_embed(self, title, fields, color=0x3498db):
        """
        Sends a custom embed message to Discord.
        fields: list of {"name": str, "value": str, "inline": bool}
        """
        embed = {
            "title": title,
            "fields": fields,
            "color": color
        }
        self.send("", embed=embed)
