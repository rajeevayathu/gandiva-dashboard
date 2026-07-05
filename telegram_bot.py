"""
Gandiva Scanner Bot — Telegram integration.

Two responsibilities:
  1. send_message(chat_id, text)  — send any Markdown text to a chat
  2. get_updates()                — poll for new messages (used once to get chat_id)
  3. webhook_loop()               — lightweight polling loop that responds to /start
                                   and registers the user's chat_id automatically.
"""
import json, os, time, urllib.request, urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_ID_FILE = os.path.join(BASE_DIR, 'telegram_chat_id.json')

def _token():
    env_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    return line.split('=', 1)[1].strip()
    return os.environ.get('TELEGRAM_BOT_TOKEN', '')


def _api(method, payload=None):
    token = _token()
    if not token:
        raise RuntimeError('TELEGRAM_BOT_TOKEN not set in .env')
    url  = f'https://api.telegram.org/bot{token}/{method}'
    data = json.dumps(payload or {}).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def send_message(chat_id, text, parse_mode='Markdown'):
    """Send a text message. Falls back to plain text if Markdown parse fails."""
    try:
        return _api('sendMessage', {
            'chat_id':    chat_id,
            'text':       text,
            'parse_mode': parse_mode,
        })
    except Exception:
        # Strip Markdown symbols and retry as plain text
        plain = text.replace('*','').replace('_','').replace('`','').replace('[','').replace(']','')
        return _api('sendMessage', {'chat_id': chat_id, 'text': plain})


def send_html(chat_id, html_text):
    """Send a message with HTML formatting."""
    return _api('sendMessage', {
        'chat_id':    chat_id,
        'text':       html_text,
        'parse_mode': 'HTML',
    })


def get_updates(offset=None):
    payload = {'timeout': 0, 'limit': 50}
    if offset:
        payload['offset'] = offset
    return _api('getUpdates', payload)


def save_chat_id(chat_id):
    with open(CHAT_ID_FILE, 'w') as f:
        json.dump({'chat_id': chat_id}, f)


def load_chat_id():
    if os.path.exists(CHAT_ID_FILE):
        with open(CHAT_ID_FILE) as f:
            return json.load(f).get('chat_id')
    return None


def register_via_polling(timeout_sec=120):
    """
    Poll for the first /start message, save the sender's chat_id, and return it.
    Used once to register the user — after that chat_id is persisted.
    """
    print(f'  Waiting for you to send /start to @GandivaScannerBot (timeout {timeout_sec}s)...')
    offset   = None
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = get_updates(offset)
            for upd in resp.get('result', []):
                offset = upd['update_id'] + 1
                msg    = upd.get('message', {})
                text   = (msg.get('text') or '').strip()
                if text.startswith('/start'):
                    chat_id = msg['chat']['id']
                    save_chat_id(chat_id)
                    print(f'  Registered chat_id: {chat_id}')
                    send_message(chat_id,
                        '*Gandiva Scanner Bot activated!* \U0001F3F9\n\n'
                        'I will send you morning briefs with top VCP setups, '
                        'regime analysis, and Minervini-validated trade ideas.\n\n'
                        'Use the *Morning Brief* button in your dashboard to trigger manually.'
                    )
                    return chat_id
        except Exception as e:
            print(f'  [poll error] {e}')
        time.sleep(2)
    print('  Timeout — no /start received.')
    return None


if __name__ == '__main__':
    # Run once to register: python3 telegram_bot.py
    chat_id = load_chat_id()
    if chat_id:
        print(f'Already registered: chat_id={chat_id}')
        send_message(chat_id, '*Gandiva is online.* \U0001F3F9 Your bot is working!')
    else:
        register_via_polling()
