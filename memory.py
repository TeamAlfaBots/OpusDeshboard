"""
Two-layer memory on MongoDB.

Layer A - recent conversation (`memory` collection, TTL on `ts`): last messages of one user in one chat.
Layer B - long-term memory (`long_memory` collection, NO TTL): stable facts the user told us.
        Isolated per (chat_id, user_id), deduplicated by key, bounded in size, and filtered so that
        secrets / private contact details / sensitive categories are never stored.

Every database call goes through `_run()`: bounded concurrency + timeout + error handling, so a
MongoDB outage never crashes a handler - the bot keeps chatting without memory (degraded mode).
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from config import (
    BLOCK,
    CHAT_LOG,
    DB_CONCURRENCY,
    DB_OP_TIMEOUT,
    HISTORY_LIMIT,
    HINGLISH,
    LOG_DAYS,
    LONG_MEMORY_FACT_CHARS,
    LONG_MEMORY_IN_PROMPT,
    LONG_MEMORY_MAX_FACTS,
    LONG_MEMORY_PROMPT_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_MESSAGE_CHARS,
    MEMORY_DAYS,
    MEMORY_EXTRACT_EVERY,
    MEMORY_EXTRACT_MAX_TOKENS,
    MEMORY_EXTRACT_MSGS,
    STOP,
)
from util import BoundedDict

log = logging.getLogger("girl-chatbot.memory")


def now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ safety filter for long-term facts
_SECRET_RE = re.compile(
    r"(otp|one[- ]?time|password|passwd|\bpwd\b|passcode|\bcvv\b|\bpin\b|api[ _-]?key|secret|token|"
    r"session[ _-]?string|private[ _-]?key|seed phrase|card number|aadhaar|aadhar|\bpan\b)",
    re.I,
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"(https?://|www\.|t\.me/)", re.I)
_HANDLE_RE = re.compile(r"(?<!\w)@\w{4,}")
_DIGITS_RE = re.compile(r"(?:\d[\s\-().]*){9,}")            # phone / card / id numbers
_ADDRESS_RE = re.compile(r"(address|ghar ka pata|\bpata\b|house no|flat no|pin ?code|pincode|\bphone\b|mobile number|whatsapp number)", re.I)
_SENSITIVE_RE = re.compile(
    r"\b(depress\w*|anxiety|diagnos\w*|therapy|therapist|medication|cancer|\bhiv\b|pregnan\w*|suicid\w*|"
    r"self[- ]?harm|khudkushi|rape\w*|abus\w*|\bsex\w*|porn\w*|\bgay\b|lesbian|bisexual|transgender|"
    r"religio\w*|muslim|hindu|christian|sikh|caste|dalit|vote|voting|\bbjp\b|congress|salary|income|"
    r"loan|debt|visa|immigra\w*|arrest\w*|police|\bjail\b|criminal)\b",
    re.I,
)
_MINOR_RE = re.compile(r"(years? old|saal (?:ka|ki)|\bage\b|\bumar\b|\bclass \d+|\bschool\b|\d+(?:th|st|nd|rd) (?:class|grade))", re.I)


def is_safe_fact(text: str) -> "tuple[bool, str]":
    """(ok, reason). Rejects secrets, contact details, sensitive categories and age/minor info."""
    t = (text or "").strip()
    if len(t) < 4:
        return False, "too short"
    if len(t) > LONG_MEMORY_FACT_CHARS:
        return False, "too long"
    if _SECRET_RE.search(t):
        return False, "secret"
    if _EMAIL_RE.search(t) or _URL_RE.search(t) or _HANDLE_RE.search(t) or _DIGITS_RE.search(t) or _ADDRESS_RE.search(t):
        return False, "contact"
    if _SENSITIVE_RE.search(t):
        return False, "sensitive"
    if _MINOR_RE.search(t):
        return False, "age"
    return True, ""


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f ]+", "", (s or "").lower()).strip()


def slugify_key(s: str) -> str:
    slug = re.sub(r"[^a-z0-9\u0900-\u097f]+", "_", (s or "").lower()).strip("_")
    return slug[:40]


def derive_key(fact: str) -> str:
    words = [w for w in norm_text(fact).split() if w not in STOP][:5]
    return slugify_key("_".join(words)) or slugify_key(fact)


_TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097f]{3,}")
_COMMON = set(STOP) | set(HINGLISH) | set(BLOCK) | {
    "user", "the", "and", "for", "with", "that", "this", "have", "has", "like", "likes", "about", "from",
    "mujhe", "tumhe", "kuch", "bahut", "hai", "hain", "wala", "wali", "karta", "karti", "pasand",
}


def tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _COMMON}


def score_fact(query_tokens: set, fact: str) -> float:
    ft = tokens(fact)
    if not ft or not query_tokens:
        return 0.0
    overlap = len(ft & query_tokens)
    return overlap / (len(ft) ** 0.5) if overlap else 0.0


# ------------------------------------------------------------------ fact extraction (AI based, parsed defensively)
EXTRACTION_PROMPT = """Tum ek memory extractor ho. Neeche ek chat hai. Sirf wo STABLE facts nikalo jo USER ne khud apne baare me bataye:
preferences, interests, goals, projects/learning, naam/nickname jo wo bulwana chahta hai.

SKIP karo: temporary mood, ek baar ka sawal, assistant ki baatein, kisi aur insaan ke private details,
passwords/OTP/API keys/secrets, phone/email/address/links/usernames, health/religion/sexual/political/paise-salary/crime
details, age ya school/class, ya koi bhi sensitive baat.

Output SIRF valid JSON (koi extra text nahi):
{"facts":[{"key":"short_snake_case_key","fact":"short third-person fact","category":"preference|interest|goal|project|name|other"}]}
- fact chhota rakho (max 150 characters), user ke shabdon ke aas-paas.
- Agar naya fact kisi EXISTING key ko update karta hai to wahi key use karo.
- Kuch bhi save karne layak nahi to {"facts":[]} do. Maximum 5 facts."""

_CATEGORIES = {"preference", "interest", "goal", "project", "name", "other"}
_EXPLICIT_RE = re.compile(
    r"(yaad\s*rakh|yad\s*rakh|याद\s*रख|remember\s+(?:that|this|me)|note\s*kar|save\s*kar|dhyan\s*rakh|mera\s+naam|my\s+name\s+is)",
    re.I,
)


def parse_extraction(text: "str | None") -> list[dict]:
    """Parse the extractor's reply. Malformed JSON / wrong shapes -> [] (never raises)."""
    if not text or not isinstance(text, str):
        return []
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(t[start : end + 1])
    except (ValueError, TypeError):
        return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    out = []
    for item in facts[:5]:
        if isinstance(item, str):
            item = {"fact": item}
        if not isinstance(item, dict) or not isinstance(item.get("fact"), str):
            continue
        key = item.get("key") if isinstance(item.get("key"), str) else None
        cat = item.get("category") if item.get("category") in _CATEGORIES else "other"
        out.append({"fact": " ".join(item["fact"].split()), "key": key, "category": cat})
    return out


# ------------------------------------------------------------------ chat log for the owner dashboard
# Everything is masked BEFORE it is stored, so the database never holds raw phone numbers, e-mails, OTPs or keys.
_MASKS = [
    (re.compile(r"(?i)\b(otp|code|pin|cvv|passcode)\b(\s*(?:is|hai|h|:|-)?\s*)(\d{3,8})\b"), r"\1\2••••"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:sk|gsk|nvapi|xai)[-_][A-Za-z0-9_\-]{12,}"), "[key]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{12,}"), "[key]"),
    (re.compile(r"\+?\d(?:[\s\-().]?\d){8,}"), "[number]"),     # 9+ digits (phone / card / id), keeps surrounding spaces
]


def mask_pii(text: "str | None") -> str:
    out = (text or "")[:MAX_MESSAGE_CHARS]
    for pattern, repl in _MASKS:
        out = pattern.sub(repl, out)
    return out


# keyword flags help the owner spot conversations worth reading. Approximate on purpose (no AI call, no cost).
FLAG_PATTERNS = {
    "real?": re.compile(
        r"(real\s*(ho|hai|h|ladki|insaan|human|girl|me|mein)\b|are you (a )?(real|bot|human|ai|robot)|\bbot\s*(ho|hai|h)\b|"
        r"\bai\s*(ho|hai|h)\b|robot|fake\s*(ho|hai|h|id|account)|insaan\s*ho|sach\s*(me|mein)\s*(ladki|ho)|ladki\s*ho\s*ya|"
        r"chat\s*gpt|chatgpt|असली|रियल|बॉट)", re.I),
    "meet": re.compile(
        r"(milna|milte|milen|mil\s*sakte|\bmeet\b|video\s*call|\bvc\b|voice\s*call|call\s*kar|number\s*(do|de|dena|bhej|share)|"
        r"whatsapp|\binsta\b|instagram|\bsnap\b|snapchat|(photo|pic|selfie)\s*(bhej|send|do|dikha)|\bselfie\b|\baddress\b|"
        r"kah?an\s*rehti|\blocation\b|ghar\s*(aao|aa)|\bhotel\b|\bnude\b)", re.I),
    "love": re.compile(
        r"(love\s*(you|u|ya)\b|\bluv\b|i\s*love|pyar|pyaar|miss\s*(you|u)\b|shaadi|shadi|marry|girlfriend|\bgf\b|propose|"
        r"like you|\bjaan\b|\bbaby\b|babu|sweetheart|\bkiss\b|\bhug\b|\bcute\b|\bhot\b|\bsexy\b|beautiful|sundar|handsome|"
        r"\bcrush\b|flirt|\bdate\b)", re.I),
    "money": re.compile(
        r"(paise|\bmoney\b|recharge|\bupi\b|gpay|phonepe|paytm|gift\s*(card|bhej)|send\s*money|payment|crypto|bitcoin|"
        r"\bloan\b|udhar|\brs\.?\s*\d+|₹\s*\d+)", re.I),
    "minor?": re.compile(
        r"(\b(1[0-7]|[5-9])\s*(years?|yrs?|saal|sal)\b|\bage\s*(is\s*)?(1[0-7]|[5-9])\b|\bclass\s*(\d|1[0-2])\b|"
        r"\b\d{1,2}(th|st|nd|rd)\s*(class|grade|std)\b|\bschool\b|\bumar\s*(1[0-7]|[5-9])\b)", re.I),
    "distress": re.compile(
        r"(suicid\w*|kill myself|end my life|self[- ]?harm|khudkushi|mar jaunga|marna chah\w*|jeena nahi|jina nahi|"
        r"मर जाऊं|आत्महत्या)", re.I),
}
FLAGS = tuple(FLAG_PATTERNS)
_BOT_SIDE_FLAGS = {"love"}          # these also look at the bot's own reply (flirty replies)


def detect_flags(user_text: "str | None", bot_text: "str | None" = None) -> list[str]:
    u, b = user_text or "", bot_text or ""
    found = []
    for name, pattern in FLAG_PATTERNS.items():
        if pattern.search(u) or (name in _BOT_SIDE_FLAGS and pattern.search(b)):
            found.append(name)
    return found


def _iso(dt) -> "str | None":
    dt = _utc(dt)
    return dt.isoformat() if isinstance(dt, datetime) else None


def _utc(dt):
    """MongoDB hands back naive UTC datetimes unless tz_aware is set; normalise to aware UTC."""
    if isinstance(dt, datetime) and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ------------------------------------------------------------------ the memory service
class Memory:
    def __init__(self, db):
        self.db = db
        self.ok = True               # False while MongoDB calls are failing (health shows "degraded")
        self.last_error = ""
        self._sem = None
        self._sem_loop = None
        self._last_ts = None               # last timestamp handed out (keeps message order strictly increasing)
        self._counts = BoundedDict(5000)   # per (chat, user) message counter for extraction cadence
        self._extract_sem = None
        self._extract_loop = None

    # ---- plumbing
    def _get_sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(DB_CONCURRENCY), loop
        return self._sem

    def _get_extract_sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._extract_sem is None or self._extract_loop is not loop:
            self._extract_sem, self._extract_loop = asyncio.Semaphore(1), loop
        return self._extract_sem

    async def _run(self, factory, default, what: str):
        """Run one DB operation with concurrency limit + timeout; on failure log and return `default`."""
        try:
            async with self._get_sem():
                result = await asyncio.wait_for(factory(), DB_OP_TIMEOUT)
            self.ok = True
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ok = False
            self.last_error = type(exc).__name__
            log.warning("MongoDB %s failed (%s) - continuing without it", what, type(exc).__name__)
            return default

    async def ping(self) -> None:
        await asyncio.wait_for(self.db.command("ping"), DB_OP_TIMEOUT)

    async def _ttl_index(self, collection: str, field: str, seconds: int) -> None:
        col = getattr(self.db, collection)
        try:
            await col.create_index(field, expireAfterSeconds=seconds)
        except Exception:
            # an older TTL index with another expireAfterSeconds exists -> update it in place
            await self.db.command("collMod", collection, index={"keyPattern": {field: 1}, "expireAfterSeconds": seconds})

    async def ensure_indexes(self) -> bool:
        async def run():
            await self.db.memory.create_index([("chat_id", 1), ("user_id", 1), ("ts", -1)])
            await self._ttl_index("memory", "ts", MEMORY_DAYS * 86400)          # recent memory only
            # long-term memory: unique key per user/chat, and deliberately NO TTL index
            await self.db.long_memory.create_index(
                [("chat_id", 1), ("user_id", 1), ("key", 1)], unique=True
            )
            await self.db.long_memory.create_index([("chat_id", 1), ("user_id", 1), ("created", 1)])
            if CHAT_LOG:                                                         # owner dashboard data
                await self.db.chat_log.create_index([("chat_id", 1), ("user_id", 1), ("ts", -1)])
                await self._ttl_index("chat_log", "ts", LOG_DAYS * 86400)
                await self._ttl_index("chat_users", "last_ts", LOG_DAYS * 86400)
            return True

        return bool(await self._run(run, False, "index setup"))

    # ---- chat settings / users
    async def get_enabled(self, chat_id: int) -> "bool | None":
        """True/False if stored, None if unknown (DB down) - caller falls back to its cache/default."""
        failed = object()
        doc = await self._run(lambda: self.db.settings.find_one({"_id": chat_id}), failed, "settings read")
        if doc is failed:
            return None
        return doc.get("enabled", True) if doc else True

    async def set_enabled(self, chat_id: int, value: bool) -> bool:
        async def run():
            await self.db.settings.update_one(
                {"_id": chat_id}, {"$set": {"enabled": value, "updated": now()}}, upsert=True
            )
            return True

        return bool(await self._run(run, False, "settings write"))

    async def touch_user(self, user) -> None:
        async def run():
            await self.db.users.update_one(
                {"_id": user.id},
                {
                    "$set": {"name": user.first_name, "username": user.username, "last_seen": now()},
                    "$setOnInsert": {"first_seen": now()},
                },
                upsert=True,
            )

        await self._run(run, None, "user update")

    # ---- layer A: recent conversation
    async def load_history(self, chat_id: int, user_id: int) -> list[dict]:
        if HISTORY_LIMIT <= 0:
            return []

        async def run():
            cur = (
                self.db.memory.find({"chat_id": chat_id, "user_id": user_id})
                .sort("ts", -1)
                .limit(HISTORY_LIMIT)
            )
            return await cur.to_list(length=HISTORY_LIMIT)

        docs = await self._run(run, [], "history read")
        msgs = []
        for d in reversed(docs or []):                      # oldest -> newest
            role, content = d.get("role"), d.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
                continue                                    # malformed / empty rows
            msgs.append({"role": role, "content": content.strip()[:MAX_MESSAGE_CHARS]})
        while msgs and msgs[0]["role"] == "assistant":      # context should start with the user
            msgs.pop(0)
        total = sum(len(m["content"]) for m in msgs)
        while msgs and total > MAX_CONTEXT_CHARS:           # bounded context size: drop the oldest
            total -= len(msgs.pop(0)["content"])
        return msgs

    async def recent_user_messages(self, chat_id: int, user_id: int, limit: int) -> list[str]:
        async def run():
            cur = (
                self.db.memory.find({"chat_id": chat_id, "user_id": user_id, "role": "user"}, {"content": 1})
                .sort("ts", -1)
                .limit(limit)
            )
            return await cur.to_list(length=limit)

        docs = await self._run(run, [], "style read")
        return [d["content"] for d in docs or [] if isinstance(d.get("content"), str) and d["content"].strip()]

    def _next_ts(self) -> datetime:
        """Timestamp for a new turn: always later than the previous turn's assistant message, so a quick
        second turn can never sort in front of the first one (MongoDB stores ms precision)."""
        t = now()
        if self._last_ts is not None and t <= self._last_ts:
            t = self._last_ts + timedelta(milliseconds=1)
        self._last_ts = t + timedelta(milliseconds=1)          # the assistant message of this turn
        return t

    async def save_turn(self, chat_id: int, user_id: int, user_text: str, bot_text: str) -> bool:
        user_text = (user_text or "").strip()[:MAX_MESSAGE_CHARS]
        bot_text = (bot_text or "").strip()[:MAX_MESSAGE_CHARS]
        if not user_text or not bot_text:
            return False
        t = self._next_ts()

        async def run():
            await self.db.memory.insert_many(
                [
                    {"chat_id": chat_id, "user_id": user_id, "role": "user", "content": user_text, "ts": t},
                    {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "role": "assistant",
                        "content": bot_text,
                        "ts": t + timedelta(milliseconds=1),
                    },
                ],
                ordered=True,
            )
            return True

        return bool(await self._run(run, False, "history write"))

    async def clear_history(self, chat_id: int, user_id: int) -> None:
        """Delete the recent conversation of one user in one chat (used by .forgetall)."""
        await self._run(
            lambda: self.db.memory.delete_many({"chat_id": chat_id, "user_id": user_id}), None, "history clear"
        )

    # ---- owner dashboard: chat log (masked, TTL, separate from memory)
    async def log_turn(self, *, chat_id: int, chat_type: str, chat_title: "str | None", user_id: int,
                       name: "str | None", username: "str | None", user_text: str, bot_text: "str | None",
                       status: str = "ok", kind: str = "text") -> None:
        """Best effort: never raises, never blocks the reply path for long."""
        if not CHAT_LOG:
            return
        u_text, b_text = mask_pii(user_text), mask_pii(bot_text) if bot_text else None
        flags = detect_flags(u_text, b_text)
        t = now()
        doc = {
            "chat_id": chat_id, "chat_type": chat_type, "chat_title": (chat_title or None),
            "user_id": user_id, "name": name, "username": username, "ts": t,
            "user_text": u_text, "bot_text": b_text, "status": status, "kind": kind, "flags": flags,
        }

        async def run():
            await self.db.chat_log.insert_one(doc)
            await self.db.chat_users.update_one(
                {"_id": f"{chat_id}:{user_id}"},
                {
                    "$set": {"chat_id": chat_id, "user_id": user_id, "name": name, "username": username,
                             "chat_title": chat_title or None, "chat_type": chat_type, "last_ts": t,
                             "last_text": u_text[:120]},
                    "$inc": {"count": 1},
                    "$addToSet": {"flags": {"$each": flags}},
                },
                upsert=True,
            )

        await self._run(run, None, "chat log write")

    async def clear_log(self, chat_id: int, user_id: int) -> None:
        """Used by .forgetall: the user asked to be forgotten, so the dashboard log goes too."""
        async def run():
            await self.db.chat_log.delete_many({"chat_id": chat_id, "user_id": user_id})
            await self.db.chat_users.delete_one({"_id": f"{chat_id}:{user_id}"})

        await self._run(run, None, "chat log clear")

    async def dash_users(self, q: str = "", flag: str = "", limit: int = 40) -> "list[dict] | None":
        """Users (per chat) ordered by latest activity. None = database unavailable."""
        flt: dict = {}
        if flag in FLAGS:
            flt["flags"] = flag
        if q:
            rx = {"$regex": re.escape(q[:100]), "$options": "i"}
            flt["$or"] = [{"name": rx}, {"username": rx}, {"chat_title": rx}]
        failed = object()

        async def run():
            return await self.db.chat_users.find(flt).sort("last_ts", -1).limit(limit).to_list(length=limit)

        docs = await self._run(run, failed, "dashboard users")
        if docs is failed:
            return None
        return [
            {"chat_id": d.get("chat_id"), "user_id": d.get("user_id"), "name": d.get("name"),
             "username": d.get("username"), "chat_title": d.get("chat_title"), "chat_type": d.get("chat_type"),
             "count": d.get("count", 0), "last_ts": _iso(d.get("last_ts")), "last_text": d.get("last_text", ""),
             "flags": list(d.get("flags") or [])}
            for d in docs
        ]

    async def dash_messages(self, *, chat_id: "int | None" = None, user_id: "int | None" = None, q: str = "",
                            flag: str = "", before: "datetime | None" = None, limit: int = 40):
        """Newest first. Returns (messages, next_before) or None when the database is unavailable."""
        flt: dict = {}
        if chat_id is not None:
            flt["chat_id"] = chat_id
        if user_id is not None:
            flt["user_id"] = user_id
        if flag in FLAGS:
            flt["flags"] = flag
        if q:
            rx = {"$regex": re.escape(q[:100]), "$options": "i"}
            flt["$or"] = [{"user_text": rx}, {"bot_text": rx}]
        if before is not None:
            flt["ts"] = {"$lt": before}
        failed = object()

        async def run():
            return await self.db.chat_log.find(flt).sort("ts", -1).limit(limit).to_list(length=limit)

        docs = await self._run(run, failed, "dashboard messages")
        if docs is failed:
            return None
        out = [
            {"ts": _iso(d.get("ts")), "chat_id": d.get("chat_id"), "user_id": d.get("user_id"), "name": d.get("name"),
             "username": d.get("username"), "chat_title": d.get("chat_title"), "chat_type": d.get("chat_type"),
             "user_text": d.get("user_text", ""), "bot_text": d.get("bot_text"), "status": d.get("status", "ok"),
             "kind": d.get("kind", "text"), "flags": list(d.get("flags") or [])}
            for d in docs
        ]
        next_before = out[-1]["ts"] if len(out) >= limit else None
        return out, next_before

    # ---- layer B: long-term facts
    async def list_facts(self, chat_id: int, user_id: int) -> list[dict]:
        async def run():
            cur = (
                self.db.long_memory.find({"chat_id": chat_id, "user_id": user_id})
                .sort("created", 1)
                .limit(LONG_MEMORY_MAX_FACTS)
            )
            return await cur.to_list(length=LONG_MEMORY_MAX_FACTS)

        return await self._run(run, [], "facts read") or []

    async def add_fact(self, chat_id: int, user_id: int, fact: str, *, key: "str | None" = None,
                       category: str = "other", source: str = "auto") -> str:
        """Returns: added | updated | duplicate | rejected | limit | error."""
        fact = " ".join((fact or "").split())
        ok, _reason = is_safe_fact(fact)
        if not ok:
            return "rejected"
        key = slugify_key(key or "") or derive_key(fact)
        if not key:
            return "rejected"
        if category not in _CATEGORIES:
            category = "other"

        facts = await self.list_facts(chat_id, user_id)
        if not self.ok:
            return "error"
        for existing in facts:
            same_key = existing.get("key") == key
            same_text = norm_text(existing.get("fact", "")) == norm_text(fact)
            if same_text:
                return "duplicate"
            if same_key:                                    # same topic, new value -> update in place
                async def upd(existing=existing):
                    await self.db.long_memory.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"fact": fact, "category": category, "updated": now(), "source": source}},
                    )
                    return True

                return "updated" if await self._run(upd, False, "fact update") else "error"

        if len(facts) >= LONG_MEMORY_MAX_FACTS:             # make room: drop the oldest automatic fact
            auto = [f for f in facts if f.get("source") != "explicit"]
            if not auto:
                return "limit"
            oldest = auto[0]
            await self._run(lambda: self.db.long_memory.delete_one({"_id": oldest["_id"]}), None, "fact evict")

        doc = {
            "chat_id": chat_id, "user_id": user_id, "key": key, "fact": fact, "category": category,
            "source": source, "created": now(), "updated": now(),
        }

        async def ins():
            await self.db.long_memory.insert_one(doc)
            return "added"

        try:
            async with self._get_sem():
                result = await asyncio.wait_for(ins(), DB_OP_TIMEOUT)
            self.ok = True
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) == 11000 or "duplicate" in str(exc).lower():
                return "duplicate"                          # lost a race with the unique index: fine
            self.ok = False
            log.warning("MongoDB fact insert failed (%s)", type(exc).__name__)
            return "error"

    async def delete_fact(self, chat_id: int, user_id: int, index: int) -> "str | None":
        """Delete the Nth fact (1-based, same numbering as `.mymemory`). Returns its text or None."""
        facts = await self.list_facts(chat_id, user_id)
        if not 1 <= index <= len(facts):
            return None
        target = facts[index - 1]
        done = await self._run(lambda: self.db.long_memory.delete_one({"_id": target["_id"]}), False, "fact delete")
        return target["fact"] if done is not False else None

    async def clear_facts(self, chat_id: int, user_id: int) -> int:
        count = await self._run(
            lambda: self.db.long_memory.count_documents({"chat_id": chat_id, "user_id": user_id}), 0, "fact count"
        )
        await self._run(
            lambda: self.db.long_memory.delete_many({"chat_id": chat_id, "user_id": user_id}), None, "fact clear"
        )
        return int(count or 0)

    async def relevant_facts(self, chat_id: int, user_id: int, queries: list[str]) -> list[str]:
        """Only facts that overlap with the current conversation - never blindly all of them."""
        if LONG_MEMORY_IN_PROMPT <= 0:
            return []
        qt = set()
        for q in queries:
            qt |= tokens(q)
        if not qt:
            return []
        facts = await self.list_facts(chat_id, user_id)
        scored = sorted(
            ((score_fact(qt, f["fact"]), f.get("updated") or f.get("created") or now(), f["fact"]) for f in facts),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        picked, used = [], 0
        for score, _ts, fact in scored:
            if score <= 0 or len(picked) >= LONG_MEMORY_IN_PROMPT:
                break
            if used + len(fact) > LONG_MEMORY_PROMPT_CHARS:
                break
            picked.append(fact)
            used += len(fact)
        return picked

    # ---- automatic extraction
    def should_extract(self, chat_id: int, user_id: int, text: str) -> bool:
        key = (chat_id, user_id)
        if _EXPLICIT_RE.search(text or ""):
            self._counts[key] = 0
            return True
        if MEMORY_EXTRACT_EVERY <= 0:
            return False
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n
        if n >= MEMORY_EXTRACT_EVERY:
            self._counts[key] = 0
            return True
        return False

    async def extract_and_store(self, chat_id: int, user_id: int, history: list[dict], ask) -> int:
        """Ask the AI for stable facts and store the safe ones. Never raises. Returns facts stored."""
        if not history:
            return 0
        try:
            async with self._get_extract_sem():              # one extraction at a time
                existing = await self.list_facts(chat_id, user_id)
                lines = [f"{m['role']}: {m['content'][:300]}" for m in history[-MEMORY_EXTRACT_MSGS:]]
                known = "\n".join(f"{f['key']}: {f['fact']}" for f in existing[-40:]) or "(none)"
                messages = [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Existing facts (key: fact):\n{known}\n\nChat:\n" + "\n".join(lines) + "\n\nJSON:"},
                ]
                raw = await ask(messages, max_tokens=MEMORY_EXTRACT_MAX_TOKENS, temperature=0.2)
                stored = 0
                for item in parse_extraction(raw):
                    result = await self.add_fact(
                        chat_id, user_id, item["fact"], key=item["key"], category=item["category"], source="auto"
                    )
                    stored += result in ("added", "updated")
                return stored
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("memory extraction failed (%s)", type(exc).__name__)
            return 0


def format_facts_block(facts: list[str]) -> str:
    if not facts:
        return ""
    return (
        "\n\nUSER KE BAARE ME JO TUM JAANTI HO (sirf tab use karo jab baat se match kare, natural tareeke se; "
        "yaad hone ka dikhava mat karo, list mat sunao):\n" + "\n".join(f"- {f}" for f in facts)
    )
