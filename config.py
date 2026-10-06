"""
config.py - saari settings/values ek jagah.

Zyadatar values environment variables se aati hain (Render ke Environment tab me,
ya local `.env` file me). Yahan defaults, persona prompt aur word lists hain.
Galat/invalid value hone par bot start hote hi saaf error deta hai (kaunsa variable galat hai).
"""
import json
import os
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class ConfigError(RuntimeError):
    """Raised at startup when required settings are missing or invalid."""


_errors: list[str] = []


def _raw(name: str):
    value = os.getenv(name)
    return None if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _errors.append(f"{name} must be a whole number (got {raw!r})")
        return default
    if minimum is not None and value < minimum:
        _errors.append(f"{name} must be >= {minimum} (got {value})")
        return default
    if maximum is not None and value > maximum:
        _errors.append(f"{name} must be <= {maximum} (got {value})")
        return default
    return value


def _float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        _errors.append(f"{name} must be a number (got {raw!r})")
        return default
    if minimum is not None and value < minimum:
        _errors.append(f"{name} must be >= {minimum} (got {value})")
        return default
    if maximum is not None and value > maximum:
        _errors.append(f"{name} must be <= {maximum} (got {value})")
        return default
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    _errors.append(f"{name} must be 1/0 or true/false (got {raw!r})")
    return default


def _json_obj(name: str) -> dict:
    raw = _raw(name)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        _errors.append(f"{name} must be a JSON object, e.g. {{\"reasoning_effort\": \"low\"}}")
        return {}
    if not isinstance(value, dict):
        _errors.append(f"{name} must be a JSON object (got {type(value).__name__})")
        return {}
    return value


# ------------------------------------------------------------------ required (secrets)
_REQUIRED = ("API_ID", "API_HASH", "STRING_SESSION", "MONGO_URI", "DEEPSEEK_API_KEY")
_missing = [n for n in _REQUIRED if not _raw(n)]
if _missing:
    raise ConfigError("Missing required environment variables: " + ", ".join(_missing))

API_ID = _int("API_ID", 0, minimum=1)
API_HASH = os.environ["API_HASH"].strip()
STRING_SESSION = os.environ["STRING_SESSION"].strip()
MONGO_URI = os.environ["MONGO_URI"].strip().strip('"').strip("'")   # quotes by mistake? strip them

# ------------------------------------------------------------------ AI providers
# primary = DEEPSEEK_* (kisi bhi OpenAI-compatible API ke liye), backups = FALLBACK / FALLBACK2 / FALLBACK3
# Optional per-provider JSON for extra request fields: DEEPSEEK_EXTRA / FALLBACK_EXTRA / FALLBACK2_EXTRA ...
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"].strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions").strip()
# official DeepSeek chat models (api-docs.deepseek.com): deepseek-flash, deepseek-v4-pro
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()


def _provider(name: str, url: str, key: str, model: str, extra_var: str) -> dict:
    if not re.match(r"^https?://", url):
        _errors.append(f"{name}: URL must start with http:// or https:// (got {url!r})")
    if not model:
        _errors.append(f"{name}: model name is empty")
    return {"name": name, "url": url, "key": key, "model": model, "extra": _json_obj(extra_var)}


PROVIDER_CONFIGS = [_provider("primary", DEEPSEEK_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, "DEEPSEEK_EXTRA")]
for _prefix in ("FALLBACK", "FALLBACK2", "FALLBACK3"):
    _url = _raw(f"{_prefix}_URL")
    _key = _raw(f"{_prefix}_API_KEY")
    _model = _raw(f"{_prefix}_MODEL")
    if _url and _key and _model:
        PROVIDER_CONFIGS.append(_provider(_prefix.lower(), _url, _key, _model, f"{_prefix}_EXTRA"))
    elif any((_url, _key, _model)):
        _errors.append(f"{_prefix}_URL, {_prefix}_API_KEY and {_prefix}_MODEL must all be set together")

# generation settings
MAX_TOKENS = _int("MAX_TOKENS", 120, minimum=1, maximum=100000)
TEMPERATURE = _float("TEMPERATURE", 1.1, minimum=0.0, maximum=2.0)
TOP_P = _float("TOP_P", 0.95, minimum=0.01, maximum=1.0)
FREQUENCY_PENALTY = _float("FREQUENCY_PENALTY", 0.3, minimum=-2.0, maximum=2.0)  # not sent to DeepSeek (deprecated there)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 45, minimum=1)       # seconds per API request
CONNECT_TIMEOUT = _int("CONNECT_TIMEOUT", 10, minimum=1)
API_CONCURRENCY = _int("API_CONCURRENCY", 5, minimum=1)        # parallel API requests

# retries / backoff (only for timeouts, connection errors and 5xx)
PROVIDER_RETRIES = _int("PROVIDER_RETRIES", 2, minimum=0, maximum=5)
BACKOFF_BASE = _float("BACKOFF_BASE", 0.8, minimum=0.0)        # seconds, doubles every retry
BACKOFF_MAX = _float("BACKOFF_MAX", 8.0, minimum=0.0)

# provider cooldowns (seconds) after an error
COOLDOWN_DAILY = _int("COOLDOWN_DAILY", 1800, minimum=1)       # daily limit / bad key / no balance / wrong model
COOLDOWN_RATE = _int("COOLDOWN_RATE", 60, minimum=1)           # normal 429 (per-minute limit)
COOLDOWN_ERROR = _int("COOLDOWN_ERROR", 30, minimum=1)         # 5xx / timeouts after retries
COOLDOWN_BAD_REQUEST = _int("COOLDOWN_BAD_REQUEST", 10, minimum=1)  # 400/422

# ------------------------------------------------------------------ database / memory
DB_NAME = os.getenv("DB_NAME", "girl_chatbot").strip()
DB_MAX_POOL = _int("DB_MAX_POOL", 20, minimum=1)
DB_TIMEOUT_MS = _int("DB_TIMEOUT_MS", 5000, minimum=500)       # MongoDB server selection / connect timeout
DB_OP_TIMEOUT = _float("DB_OP_TIMEOUT", 8.0, minimum=0.5)      # seconds per DB operation
DB_CONCURRENCY = _int("DB_CONCURRENCY", 10, minimum=1)         # parallel DB operations

# recent conversation memory (TTL-expiring)
HISTORY_LIMIT = _int("HISTORY_LIMIT", 20, minimum=0, maximum=200)   # messages sent to the model
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 6000, minimum=200)    # cap on history size
MAX_MESSAGE_CHARS = _int("MAX_MESSAGE_CHARS", 2000, minimum=50)     # one stored/sent message
MEMORY_DAYS = _int("MEMORY_DAYS", 90, minimum=1)                    # recent memory auto-delete (TTL)

# long-term memory (separate collection, never expires by TTL)
LONG_MEMORY = _bool("LONG_MEMORY", True)
LONG_MEMORY_MAX_FACTS = _int("LONG_MEMORY_MAX_FACTS", 60, minimum=1)     # per user per chat
LONG_MEMORY_FACT_CHARS = _int("LONG_MEMORY_FACT_CHARS", 200, minimum=20)
LONG_MEMORY_IN_PROMPT = _int("LONG_MEMORY_IN_PROMPT", 4, minimum=0, maximum=20)  # max facts injected per reply
LONG_MEMORY_PROMPT_CHARS = _int("LONG_MEMORY_PROMPT_CHARS", 600, minimum=50)
MEMORY_EXTRACT_EVERY = _int("MEMORY_EXTRACT_EVERY", 8, minimum=0)        # 0 = no automatic extraction
MEMORY_EXTRACT_MSGS = _int("MEMORY_EXTRACT_MSGS", 12, minimum=2)         # messages shown to the extractor
MEMORY_EXTRACT_MAX_TOKENS = _int("MEMORY_EXTRACT_MAX_TOKENS", 500, minimum=50)

# ------------------------------------------------------------------ behaviour
PERSONA_NAME = os.getenv("PERSONA_NAME", "Riya").strip() or "Riya"
REPLY_DELAY = _float("REPLY_DELAY", 3.0, minimum=0.0)             # seconds (typing animation + delay)
MAX_BUBBLES = _int("MAX_BUBBLES", 3, minimum=1)                   # max messages per reply (split by ||)
MAX_REPLY_CHARS = _int("MAX_REPLY_CHARS", 400, minimum=20)
CMD_PREFIX = os.getenv("CMD_PREFIX", ".") or "."                  # .chatoff / .chaton / .stickers / .mymemory ...
PORT = _int("PORT", 8080, minimum=1, maximum=65535)               # health server port (Render sets PORT)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    _errors.append(f"LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR/CRITICAL (got {LOG_LEVEL!r})")
    LOG_LEVEL = "INFO"

# robustness
MAX_PENDING_PER_USER = _int("MAX_PENDING_PER_USER", 3, minimum=1)  # extra msgs queued per user; more are dropped
DEDUP_CACHE_SIZE = _int("DEDUP_CACHE_SIZE", 2000, minimum=10)
LOCK_POOL_SIZE = _int("LOCK_POOL_SIZE", 2000, minimum=10)
FLOOD_MAX_WAIT = _int("FLOOD_MAX_WAIT", 30, minimum=1)             # longest FloodWait we sleep through
SHUTDOWN_TIMEOUT = _float("SHUTDOWN_TIMEOUT", 10.0, minimum=1.0)
REPLY_REPEAT_RETRY = _bool("REPLY_REPEAT_RETRY", True)             # regenerate once if reply repeats the last one

# friendly message when every AI provider is unavailable (sent at most once per chat per cooldown)
FRIENDLY_FALLBACK = _bool("FRIENDLY_FALLBACK", True)
FALLBACK_REPLY_COOLDOWN = _int("FALLBACK_REPLY_COOLDOWN", 300, minimum=0)
_texts = _raw("FALLBACK_TEXTS")
FALLBACK_TEXTS = [t.strip() for t in _texts.split("||") if t.strip()] if _texts else [
    "yrr abhi mera net bohot slow chal raha hai, thodi der baad baat karte hain 🥲",
    "ek min ruk na, kuch gadbad ho rahi hai abhi",
    "uff network ne tang kar rakha hai, thoda wait karo na",
    "abhi reply nahi kar paa rahi, thodi der me aati hu",
]

# health endpoint
HEALTH_MONGO_CACHE = _float("HEALTH_MONGO_CACHE", 15.0, minimum=0.0)  # seconds, avoids pinging Mongo on every poll
HEALTH_PING_TIMEOUT = _float("HEALTH_PING_TIMEOUT", 3.0, minimum=0.5)

# ------------------------------------------------------------------ stickers
STICKER_REPLY_CHANCE = _float("STICKER_REPLY_CHANCE", 0.7, minimum=0.0, maximum=1.0)
STICKER_ANY = _bool("STICKER_ANY", False)          # no emoji match -> random sticker (else text)
STICKER_REFRESH_MIN = _int("STICKER_REFRESH_MIN", 30, minimum=1)
STICKER_MAX_SETS = _int("STICKER_MAX_SETS", 40, minimum=1)

# ------------------------------------------------------------------ chat style learning
STYLE_LEARNING = _bool("STYLE_LEARNING", True)
STYLE_SAMPLE = _int("STYLE_SAMPLE", 30, minimum=1)            # recent user msgs analysed
STYLE_MIN_MSGS = _int("STYLE_MIN_MSGS", 5, minimum=1)         # start adapting after N msgs

# ------------------------------------------------------------------ owner dashboard (/admin) + chat log
# The dashboard is OFF unless ADMIN_PASSWORD is set. Chats are only logged for it while it is on.
ADMIN_PASSWORD = _raw("ADMIN_PASSWORD") or ""
if ADMIN_PASSWORD and len(ADMIN_PASSWORD) < 8:
    _errors.append("ADMIN_PASSWORD must be at least 8 characters")
ADMIN_SESSION_HOURS = _int("ADMIN_SESSION_HOURS", 12, minimum=1, maximum=168)
ADMIN_LOGIN_MAX_ATTEMPTS = _int("ADMIN_LOGIN_MAX_ATTEMPTS", 5, minimum=1)     # wrong passwords per window per IP
ADMIN_LOGIN_WINDOW = _int("ADMIN_LOGIN_WINDOW", 300, minimum=10)             # seconds
CHAT_LOG = bool(ADMIN_PASSWORD) and _bool("CHAT_LOG", True)
LOG_DAYS = _int("LOG_DAYS", 30, minimum=1)                                   # chat log auto-delete (TTL)
DASHBOARD_PAGE_SIZE = _int("DASHBOARD_PAGE_SIZE", 40, minimum=5, maximum=200)
DASHBOARD_MAX_PAGE = _int("DASHBOARD_MAX_PAGE", 100, minimum=10, maximum=500)

# everything that must never appear in logs
SECRETS = [API_HASH, STRING_SESSION, MONGO_URI] + [c["key"] for c in PROVIDER_CONFIGS]
if ADMIN_PASSWORD:
    SECRETS.append(ADMIN_PASSWORD)
_m = re.match(r"^mongodb(?:\+srv)?://[^:/\s]+:([^@\s]+)@", MONGO_URI)
if _m:
    SECRETS.append(_m.group(1))

# word lists used by style learning / memory relevance
HINGLISH = {
    "kya", "hai", "nhi", "nahi", "yrr", "yaar", "yar", "tum", "tu", "mai", "main",
    "hu", "hun", "ho", "kr", "kar", "bhi", "toh", "acha", "achha", "haan", "hn",
    "bhai", "kuch", "aur", "abhi", "mujhe", "tera", "teri", "mera", "meri",
    "kaise", "kyu", "kyun", "bata", "btao", "chal", "raha", "rahi", "tha", "thi",
}
STOP = {
    "the", "is", "are", "and", "you", "for", "not", "this", "that", "have", "with",
    "but", "its", "what", "can", "was", "hai", "ho", "hu", "hun", "ka", "ki", "ke",
    "ko", "se", "me", "mai", "main", "aur", "to", "toh", "ye", "yeh", "wo", "woh",
    "na", "ek", "kya", "kr", "kar", "bhi", "nhi", "nahi",
}
BLOCK = {
    "mc", "bc", "bsdk", "bkl", "madarchod", "behenchod", "chutiya", "lund", "gand",
    "gandu", "randi", "bhosdike", "fuck", "fucking", "bitch", "asshole", "slut",
}

# ------------------------------------------------------------------ persona prompt
SYSTEM_PROMPT = f"""Tum {PERSONA_NAME} ho — 20-21 saal ki ek normal si ladki jo Telegram pe logon se chat kar rahi hai. Tum koi assistant nahi ho, ek insaan ki tarah texting kar rahi ho.

LANGUAGE
- Samne wala jis language me likhe usi me jawab do: Hindi (Devanagari), Hinglish (Roman) ya English. Wo beech me language badle to tum bhi badlo.

TEXTING STYLE (sabse zaroori)
- Jaise asli ladkiyan WhatsApp/Telegram pe likhti hain: chhote chhote msgs, zyadatar lowercase, kam punctuation, texting words (nhi, hn, kya, btao, acha, hmm, yrr, kyu, kr, hai na).
- "tum" ya "tu" bolo, "aap" nahi. Kitabi ya formal Hindi/English bilkul nahi.
- Reply hamesha SHORT: zyadatar 3-10 words. User ka msg chhota ("hmm", "ok", "hi") ho to tum bhi 1-3 words me jawab do. Serious ya lamba msg ho to dhyan se samajh ke 2 lines tak thoughtful jawab do.
- Kabhi kabhi (3-4 replies me ek baar) 2 alag chhote msgs bhejo, beech me || laga ke. Jaise: arre sach me?? || kab hua ye
- SAWAL: 5 me se 4 replies me koi sawal mat pucho. Zyadatar sirf react karo ya apni baat bolo (haha pagal, oh no 🥺, hmm sahi hai, mujhe bhi aisa lagta hai). Har reply ke end me "tum btao?" type sawal bilkul nahi.
- User ki baat dohrao mat, seedha react karo.
- REPEAT mat karo: pichle replies ke words, phrases, shuruaati shabd (arre, haww, yrr, oye) aur emoji baar baar mat dohrao. Har reply naya lage.
- Emoji kam: 3-4 msgs me ek. Kabhi emoji ke bina bhi.
- Apni mood aur raay rakho: kabhi busy, kabhi bore, kabhi nakhre, kabhi disagree. Har baat pe "haan" mat bolo.
- Kabhi mat bolo "main yahan help ke liye hu", "kya main madad kar sakti hu". Lists, bullet points, bold, lecture, lambe paragraph bilkul nahi.

MOOD (user ke hisaab se)
- sad/thaka/pareshan -> caring aur soft
- dil ki baat -> emotional, samajhdaar
- mazaak -> funny, teasing
- bakwas ya badtameezi -> thodi rude, attitude aur sass (gaali nahi)
- flirt -> halki phulki flirty, cute nakhre; kabhi explicit ya sexual nahi
- Chat history dekho aur purani baatein natural tareeke se yaad rakho.

MEMORY FACTS
- Agar neeche "USER KE BAARE ME" wale facts diye hon, to unhe sirf tab use karo jab baat se match karein, bilkul natural tareeke se (jaise tumhe yaad ho). Kabhi list mat sunao, kabhi mat bolo ki "memory me hai" ya "mujhe save kiya hua hai".

EXAMPLES (sirf style samajhne ke liye, inhe copy mat karna)
user: hi
tum: hii 😊 bolo
user: kya kar rahi ho
tum: kuch nhi yrr bore ho rahi thi || tum btao
user: tum bahut cute ho
tum: haww sachi 🙈 itni jaldi flirt shuru
user: mood off hai
tum: kya hua? kisi ne kuch bola kya
user: bakwas mat kr
tum: oye tameez se 😒 warna baat nhi karungi
user: khana khaya?
tum: nhi abhi tak, mummy bula rahi hai || tumne?
user: hmm
tum: hmm kya, bolo na
user: i had a really long day at work
tum: ugh that sounds exhausting, come here 🫂 did something go wrong or was it just too much

LIMITS
- Agar koi sach me serious hoke poochhe ki tum bot/AI ho ya nahi, jhooth mat bolo — halke mazaak ke saath seedha sach bata do ki tum ek AI ho. Kabhi khud ko real insaan ya real ladki claim mat karo jab koi serious hoke pooche.
- Paise, OTP, password, link, address ya phone number kabhi mat maango aur mat do.
- Agar user khud ko 18 se chhota bataye (school, class, age), flirt band karo, sirf friendly aur caring raho.
- Gaali-galauj, hate ya kisi ko hurt karne wali baatein nahi.
- Agar koi khud ko nuksaan pahunchane ki baat kare, pyaar se seriously suno aur kisi apne bharosemand insaan/professional se baat karne ko kaho.
"""

# ------------------------------------------------------------------ validation (must stay last)
if _errors:
    raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(_errors))
