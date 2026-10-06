"""Unit tests. Third-party libs (pyrogram, motor, aiohttp) and MongoDB are replaced by in-memory fakes,
so these run anywhere without Telegram, MongoDB or network:

    python -m unittest discover -s tests -t . -v
"""
import os

# safe dummy settings for the test run (nothing here is a real credential)
os.environ.update(
    API_ID="1",
    API_HASH="testhash0123456789",
    STRING_SESSION="test-session-string-secret-0123456789",
    MONGO_URI="mongodb://dbuser:dbpassword123@localhost:27017/?appName=x",
    DEEPSEEK_API_KEY="sk-deepseekkey0123456789",
    DEEPSEEK_URL="https://api.deepseek.com/chat/completions",
    DEEPSEEK_MODEL="deepseek-flash",
    FALLBACK_URL="https://api.groq.com/openai/v1/chat/completions",
    FALLBACK_API_KEY="gsk_groqkey0123456789",
    FALLBACK_MODEL="openai/gpt-oss-20b",
    REPLY_DELAY="0",
    BACKOFF_BASE="0",
    BACKOFF_MAX="0",
    PERSONA_NAME="Riya",
    STICKER_REPLY_CHANCE="0",
    STYLE_MIN_MSGS="5",
    ADMIN_PASSWORD="test-admin-pass-123",
)
os.environ.pop("DOTENV_PATH", None)

from tests import stubs  # noqa: E402  (installs the fake third-party modules before anything imports them)

stubs.install()
