import requests

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text: str):
        payload = {
            "chat_id": self.chat_id,
            "text": text
        }
        try:
            response = requests.post(self.api_url, data=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Errore nell'invio del messaggio: {e}")
            return None
