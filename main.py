"""
Girl-persona Telegram chat bot (single account / userbot)
Stack: PyroTGFork (Pyrogram) + OpenAI-compatible AI API (DeepSeek primary + backups) + MongoDB (motor) + aiohttp health server

Behaviour
- DM: replies to everyone. Group: only when this account is @mentioned / replied to.
- `.chatoff` / `.chaton` (sent from this account) -> off/on for THAT chat only; `.stickers` reloads your sticker packs
- Users can manage what the bot remembers about them: `.mymemory`, `.remember <text>`, `.forget <n>`, `.forgetall`
- Typing animation + REPLY_DELAY seconds delay before each reply
- Two-layer memory (memory.py): recent chat history (TTL) + separate long-term facts (no TTL)
- Provider chain with retries / cooldowns / fallback (providers.py); friendly message if every provider is down
- Health port: GET / and /health -> JSON status (health.py)
- Optional owner dashboard at /admin (dashboard.py): who wrote what and what the bot answered; needs ADMIN_PASSWORD

Settings live in config.py (env vars). Needs Python 3.10+
"""
import asyncio
import inspect
import logging
import random
import re
import sys
import time
from collections import Counter

import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters, idle, raw
from pyrogram.enums import ChatAction, ChatType
from pyrogram.errors import FloodWait
from pyrogram.file_id import FileId, FileType
from pyrogram.handlers import MessageHandler

try:
    from config import (
        API_HASH,
        API_ID,
        BLOCK,
        CHAT_LOG,
        CMD_PREFIX,
        DB_MAX_POOL,
        DB_NAME,
        DB_TIMEOUT_MS,
        DEDUP_CACHE_SIZE,
        FALLBACK_REPLY_COOLDOWN,
        FALLBACK_TEXTS,
        FLOOD_MAX_WAIT,
        FRIENDLY_FALLBACK,
        HINGLISH,
        LOCK_POOL_SIZE,
        LOG_LEVEL,
        LONG_MEMORY,
        MAX_BUBBLES,
        MAX_MESSAGE_CHARS,
        MAX_PENDING_PER_USER,
        MONGO_URI,
        PORT,
        REPLY_DELAY,
        REPLY_REPEAT_RETRY,
        SECRETS,
        SHUTDOWN_TIMEOUT,
        STICKER_ANY,
        STICKER_MAX_SETS,
        STICKER_REFRESH_MIN,
        STICKER_REPLY_CHANCE,
        STOP,
        STRING_SESSION,
        STYLE_LEARNING,
        STYLE_MIN_MSGS,
        STYLE_SAMPLE,
        SYSTEM_PROMPT,
        TEMPERATURE,
    )
except RuntimeError as exc:   # config.ConfigError: missing / invalid environment variables
    sys.stderr.write(f"\n[config error] {exc}\n")
    raise SystemExit(1)

import dashboard
import providers
from health import HealthState, start_health_server
from memory import Memory, format_facts_block
from providers import ask_deepseek
from util import BoundedDict, BoundedSet, LockPool, setup_logging

log = logging.getLogger("girl-chatbot")

# ------------------------------------------------------------------ globals (all bounded)
memory: "Memory | None" = None
state = HealthState()
_enabled_cache = BoundedDict(5000)
_locks = LockPool(LOCK_POOL_SIZE)
_pending: dict = {}                     # (chat, user) -> messages currently queued/processing
_seen = BoundedSet(DEDUP_CACHE_SIZE)    # (chat, message id) already handled
_fallback_sent = BoundedDict(2000)      # chat -> last time we sent the "AI is down" message
_bg_tasks: set = set()
_sleep = asyncio.sleep                  # indirection so tests don't really wait

USER_COMMANDS = ("mymemory", "remember", "forget", "forgetall")


def spawn(coro) -> asyncio.Task:
    """Background task that is tracked (not garbage collected) and cancelled on shutdown."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_task_done)
    return task


def _task_done(task: asyncio.Task) -> None:
    _bg_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        log.warning("background task failed (%s)", type(task.exception()).__name__)


async def _finish_task(task: asyncio.Task, timeout: float = 2.0) -> None:
    """Wait briefly for a helper task, then cancel it - never hangs the handler."""
    try:
        await asyncio.wait_for(task, timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    except Exception:
        pass


# ------------------------------------------------------------------ chat on/off
async def is_enabled(chat_id: int) -> bool:
    if chat_id in _enabled_cache:
        return _enabled_cache[chat_id]
    value = await memory.get_enabled(chat_id)
    if value is None:            # MongoDB down: default ON, don't cache the guess
        return True
    _enabled_cache[chat_id] = value
    return value


async def set_enabled(chat_id: int, value: bool) -> bool:
    """Applies immediately (cache); returns False if the DB write failed (resets on restart then)."""
    _enabled_cache[chat_id] = value
    return await memory.set_enabled(chat_id, value)


# ---- stickers: only from the sticker packs ADDED to this account ----
_sticker_cache: dict = {"ts": 0.0, "items": [], "packs": 0, "busy": False}


def norm_emoji(e: "str | None") -> str:
    return (e or "").replace("\ufe0f", "").strip()


async def load_my_stickers(client: Client, force: bool = False) -> list[dict]:
    """Read every sticker of the packs added to this account (messages.GetAllStickers)."""
    c = _sticker_cache
    fresh = c["items"] and time.monotonic() - c["ts"] < STICKER_REFRESH_MIN * 60
    if (fresh and not force) or c["busy"]:
        return c["items"]
    c["busy"] = True
    try:
        items: list[dict] = []
        all_sets = await client.invoke(raw.functions.messages.GetAllStickers(hash=0))
        sets = list(getattr(all_sets, "sets", []))[:STICKER_MAX_SETS]
        for st in sets:
            try:
                full = await client.invoke(
                    raw.functions.messages.GetStickerSet(
                        stickerset=raw.types.InputStickerSetID(id=st.id, access_hash=st.access_hash),
                        hash=0,
                    )
                )
            except FloodWait as e:
                await _sleep(min(_flood_seconds(e), 15))
                continue
            except Exception:
                log.warning("could not load sticker set %s", getattr(st, "short_name", "?"))
                continue
            for doc in getattr(full, "documents", []):
                emoji = ""
                for attr in doc.attributes:
                    if isinstance(attr, raw.types.DocumentAttributeSticker):
                        emoji = attr.alt or ""
                        break
                file_id = FileId(
                    file_type=FileType.STICKER,
                    dc_id=doc.dc_id,
                    media_id=doc.id,
                    access_hash=doc.access_hash,
                    file_reference=doc.file_reference,
                ).encode()
                items.append({"id": doc.id, "file_id": file_id, "emoji": emoji})
            await _sleep(0.2)
        if items:
            c.update(ts=time.monotonic(), items=items, packs=len(sets))
            log.info("Loaded %s stickers from %s packs", len(items), len(sets))
        else:
            log.warning("No stickers found - add sticker packs to this account")
    except Exception:
        log.warning("Loading sticker packs failed", exc_info=True)
    finally:
        c["busy"] = False
    return c["items"]


def refresh_stickers_in_background(client: Client) -> None:
    spawn(load_my_stickers(client, force=True))


async def find_my_sticker(client: Client, sticker) -> "dict | None":
    """Pick a sticker from the account's own packs that matches the user's sticker emoji."""
    c = _sticker_cache
    if time.monotonic() - c["ts"] > STICKER_REFRESH_MIN * 60 and not c["busy"]:
        refresh_stickers_in_background(client)  # serve the current list meanwhile
    items = c["items"]
    if not items:
        return None

    sent_id = None
    try:
        sent_id = FileId.decode(sticker.file_id).media_id
    except Exception:
        pass
    others = [i for i in items if i["id"] != sent_id]

    want = norm_emoji(sticker.emoji)
    matches = [i for i in others if want and norm_emoji(i["emoji"]) == want]
    if matches:
        return random.choice(matches)
    if STICKER_ANY and others:
        return random.choice(others)
    return None


# ------------------------------------------------------------------ helpers
def _flood_seconds(exc) -> int:
    for attr in ("value", "x"):
        v = getattr(exc, attr, None)
        if isinstance(v, (int, float)) and v >= 0:
            return int(v)
    return 5


async def keep_typing(client: Client, chat_id: int, stop: asyncio.Event, action=ChatAction.TYPING) -> None:
    while not stop.is_set():
        try:
            await client.send_chat_action(chat_id, action)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


def is_addressed(client: Client, message) -> bool:
    """True if this account is tagged / mentioned / replied to."""
    me = client.me
    if message.mentioned:
        return True
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == me.id:
        return True
    text = (message.text or getattr(message, "caption", None) or "").lower()
    username = (me.username or "").lower()
    if username and f"@{username}" in text:
        return True
    for ent in getattr(message, "entities", None) or []:      # "text mention" of our account (no @username)
        ent_user = getattr(ent, "user", None)
        if ent_user is not None and getattr(ent_user, "id", None) == me.id:
            return True
    return False


def strip_mention(client: Client, text: str) -> str:
    username = client.me.username
    if username:
        text = re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.I)
    return text.strip()


def is_command_text(text: str) -> bool:
    """'/start', '.chatoff' ... but not '...' or '..ok'."""
    return text.startswith("/") or bool(re.match(rf"^{re.escape(CMD_PREFIX)}[A-Za-z]", text))


# ---- chat style learning (per user, from their own recent messages) ----
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764]")
DEV_RE = re.compile("[\u0900-\u097F]")
TOKEN_RE = re.compile("[a-zA-Z\u0900-\u097F']{2,}")


def style_from_messages(msgs: list[str]) -> str:
    """Build a short style hint from a user's recent messages (pure function)."""
    n = len(msgs)
    if n < STYLE_MIN_MSGS:
        return ""
    avg_words = sum(len(m.split()) for m in msgs) / n
    emoji_rate = sum(1 for m in msgs if EMOJI_RE.search(m)) / n
    lower_rate = sum(1 for m in msgs if m == m.lower()) / n
    dev_rate = sum(1 for m in msgs if DEV_RE.search(m)) / n
    dots_rate = sum(1 for m in msgs if "..." in m or "…" in m) / n
    excl_rate = sum(1 for m in msgs if "!" in m) / n
    tokens = [t.lower() for m in msgs for t in TOKEN_RE.findall(m)]

    if dev_rate >= 0.5:
        lang = "Hindi (Devanagari script me likhta hai, tum bhi Devanagari me reply do)"
    elif tokens and sum(1 for t in tokens if t in HINGLISH) / len(tokens) >= 0.08:
        lang = "Hinglish (Roman script me Hindi+English mix)"
    else:
        lang = "English (tum bhi English me reply do)"

    if avg_words <= 4:
        length = "bahut chhote msgs (1-4 words), tum bhi 1-5 words me reply do"
    elif avg_words <= 10:
        length = "chhote msgs, tum bhi chhota rakho"
    else:
        length = "lambe msgs likhta hai, tum thoda detail me (2-3 lines tak) jawab de sakti ho"

    if emoji_rate >= 0.4:
        emoji = "emoji kaafi use karta hai, tum bhi thode zyada use karo"
    elif emoji_rate <= 0.1:
        emoji = "lagbhag emoji nahi use karta, tum bhi bahut kam (ya bilkul nahi)"
    else:
        emoji = "kabhi kabhi emoji, tum bhi kabhi kabhi"

    lines = [f"- Language: {lang}", f"- Length: {length}", f"- Emoji: {emoji}"]
    if lower_rate >= 0.8:
        lines.append("- Sab kuch lowercase me likhta hai")
    if dots_rate >= 0.25:
        lines.append("- '...' bahut lagata hai")
    if excl_rate >= 0.3:
        lines.append("- '!' bahut lagata hai, energetic style")

    common = Counter(t for t in tokens if t not in STOP and t not in BLOCK and len(t) >= 2)
    favs = [w for w, c in common.most_common(5) if c >= 3]
    if favs:
        lines.append(
            "- Aksar ye words use karta hai: " + ", ".join(favs)
            + " (kabhi kabhi tum bhi use kar sakti ho, har msg me nahi)"
        )

    return (
        "\n\nSAMNE WALE KA CHAT STYLE (isko subtly match karo jaise dost ek dusre ka "
        "style pakad lete hain; copy-paste mat karo, apni personality bani rahe; "
        "gaali ya abusive words kabhi copy mat karna):\n" + "\n".join(lines)
    )


async def build_style_hint(chat_id: int, user_id: int, current_text: str) -> str:
    if not STYLE_LEARNING:
        return ""
    docs = await memory.recent_user_messages(chat_id, user_id, STYLE_SAMPLE)
    msgs = [m for m in docs if not m.startswith("[sticker")]
    if current_text and not current_text.startswith("[sticker"):
        msgs.append(current_text)
    return style_from_messages(msgs)


def avoid_block(history: list[dict]) -> str:
    last = [m["content"] for m in history if m["role"] == "assistant"][-3:]
    if not last:
        return ""
    return "\n\nPICHLE REPLIES (inke shabd/emoji/shuruaat repeat mat karo): " + " | ".join(x[:80] for x in last)


def build_system_prompt(user, is_group: bool, style: str = "", facts_block: str = "", avoid: str = "") -> str:
    where = "ek group me (jisne tag kiya usi se baat karo)" if is_group else "DM me"
    return (
        f"{SYSTEM_PROMPT}\nAbhi tum {where} ho. Samne wale ka naam: {user.first_name or 'dost'}."
        f"{facts_block}{style}{avoid}"
    )


def norm_reply(s: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", (s or "").replace("||", " ").lower())


def is_repeat(reply: str, history: list[dict]) -> bool:
    """True if the reply is identical to one of the last 3 bot replies."""
    me = norm_reply(reply)
    recent = [norm_reply(m["content"]) for m in history if m["role"] == "assistant"][-3:]
    return bool(me) and me in recent


async def send_one(message, text: str, quote: bool) -> None:
    """Send one message; a FloodWait is waited out once (bounded), anything else propagates."""
    try:
        await message.reply_text(text, quote=quote)
    except FloodWait as e:
        wait = _flood_seconds(e)
        if wait > FLOOD_MAX_WAIT:
            raise
        log.warning("FloodWait %ss while sending, waiting", wait)
        await _sleep(wait + 1)
        await message.reply_text(text, quote=quote)


async def log_chat(message, is_group: bool, user_text: str, bot_text: "str | None", status: str = "ok", kind: str = "text") -> None:
    """Record the turn for the owner dashboard (masked, best effort, only while the dashboard is enabled)."""
    if not CHAT_LOG or memory is None:
        return
    chat, user = message.chat, message.from_user
    try:
        await memory.log_turn(
            chat_id=chat.id, chat_type="group" if is_group else "private",
            chat_title=getattr(chat, "title", None) if is_group else None,
            user_id=user.id, name=getattr(user, "first_name", None), username=getattr(user, "username", None),
            user_text=user_text, bot_text=bot_text, status=status, kind=kind,
        )
    except Exception as exc:          # logging must never break chatting
        log.warning("chat log failed (%s)", type(exc).__name__)


async def maybe_send_fallback(message, is_group: bool) -> "str | None":
    """All AI providers are down: tell the user once in a while instead of staying silent."""
    if not FRIENDLY_FALLBACK:
        return None
    chat_id = message.chat.id
    last = _fallback_sent.get(chat_id)
    if last is not None and time.monotonic() - last < FALLBACK_REPLY_COOLDOWN:
        return None
    _fallback_sent[chat_id] = time.monotonic()
    text = random.choice(FALLBACK_TEXTS)
    try:
        await send_one(message, text, quote=is_group)
        return text
    except Exception as exc:
        log.warning("could not send fallback message (%s)", type(exc).__name__)
        return None


async def generate_reply(client: Client, chat_id: int, messages: list[dict], history: list[dict]) -> "str | None":
    """AI reply produced in parallel with the typing animation + REPLY_DELAY."""
    stop = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(client, chat_id, stop))
    try:
        reply, _ = await asyncio.gather(ask_deepseek(messages), _sleep(REPLY_DELAY))
        if reply and REPLY_REPEAT_RETRY and is_repeat(reply, history):
            hinted = [dict(m) for m in messages]
            hinted[0]["content"] += "\n\nNOTE: tumhara draft pichle reply jaisa hi tha. Bilkul alag shabdon me, naye tareeke se likho."
            retry = await ask_deepseek(hinted, temperature=min(TEMPERATURE + 0.3, 1.8))
            if retry and not is_repeat(retry, history):
                reply = retry
        return reply
    finally:
        stop.set()
        await _finish_task(typing_task)


# ------------------------------------------------------------------ owner commands
async def on_command(client: Client, message) -> None:
    """`.chatoff` / `.chaton` toggle the current chat; `.stickers` reloads your sticker packs."""
    cmd = message.command[0].lower()
    try:
        if cmd == "stickers":
            items = await load_my_stickers(client, force=True)
            note = (
                f"🎴 {len(items)} stickers loaded ({_sticker_cache['packs']} packs)"
                if items
                else "⚠️ Koi sticker pack add nahi mila"
            )
        else:
            enabled = cmd == "chaton"
            saved = await set_enabled(message.chat.id, enabled)
            note = "✅ Chat bot ON (is chat me)" if enabled else "🔕 Chat bot OFF (is chat me)"
            if not saved:
                note += "\n⚠️ DB save fail hua - restart tak hi chalega"
    except Exception as exc:
        log.warning("owner command failed (%s)", type(exc).__name__)
        note = "⚠️ Command fail ho gayi"
    try:
        await message.edit_text(note)
        await _sleep(4)
        await message.delete()
    except Exception:
        pass


# ------------------------------------------------------------------ user memory commands
async def _memory_command(chat_id: int, user_id: int, cmd: str, arg: str) -> str:
    if cmd == "mymemory":
        facts = await memory.list_facts(chat_id, user_id)
        if not facts:
            if not memory.ok:
                return "abhi memory access nahi ho pa rahi, thodi der baad try karo 🥲"
            return "abhi mere paas tumhare baare me kuch save nahi hai"
        lines = [f"{i}. {f['fact'][:120]}" for i, f in enumerate(facts[:20], 1)]
        return ("Jo mujhe tumhare baare me yaad hai:\n" + "\n".join(lines)
                + f"\n\n{CMD_PREFIX}forget <number> se ek hatao, {CMD_PREFIX}forgetall se sab clear")
    if cmd == "remember":
        if not arg:
            return f"aise likho: {CMD_PREFIX}remember <jo yaad rakhna hai>"
        result = await memory.add_fact(chat_id, user_id, arg, source="explicit")
        return {
            "added": "theek hai, yaad rakh lungi ✅",
            "updated": "update kar diya ✅",
            "duplicate": "ye to pehle se yaad hai 😄",
            "rejected": "ye main save nahi kar sakti (password, number, private ya sensitive baatein save nahi hoti)",
        }.get(result, "abhi save nahi ho paya, thodi der baad try karo")
    if cmd == "forget":
        if not arg.strip().isdigit():
            return f"aise likho: {CMD_PREFIX}forget <number> ({CMD_PREFIX}mymemory me number dikhte hain)"
        removed = await memory.delete_fact(chat_id, user_id, int(arg.strip()))
        return f"hata diya ✅ ({removed[:80]})" if removed else f"ye number nahi mila, {CMD_PREFIX}mymemory dekho"
    if cmd == "forgetall":
        count = await memory.clear_facts(chat_id, user_id)
        await memory.clear_history(chat_id, user_id)
        await memory.clear_log(chat_id, user_id)
        return f"sab bhool gayi ✅ ({count} baatein + purani chat hata di)"
    return ""


async def on_user_command(client: Client, message) -> None:
    user = message.from_user
    if not user or user.is_bot:
        return
    chat = message.chat
    is_group = chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
    if is_group and not is_addressed(client, message):     # in groups: only when the bot is tagged/replied to
        return
    if not await is_enabled(chat.id):
        return
    parts = (message.text or "").split(None, 1)
    cmd = parts[0].lstrip(CMD_PREFIX).split("@")[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    if is_group:
        arg = strip_mention(client, arg)
    if cmd not in USER_COMMANDS:
        return
    answer = await _memory_command(chat.id, user.id, cmd, arg)
    if answer:
        try:
            await send_one(message, answer, quote=is_group)
        except Exception as exc:
            log.warning("could not answer memory command (%s)", type(exc).__name__)


# ------------------------------------------------------------------ stickers
async def reply_with_sticker(client: Client, message, sticker, is_group: bool, user_text: str) -> bool:
    """Reply with a sticker from this account's own packs. Returns False if not possible."""
    chat_id = message.chat.id
    pick = await find_my_sticker(client, sticker)
    if not pick:
        return False

    stop = asyncio.Event()
    action = getattr(ChatAction, "CHOOSE_STICKER", ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing(client, chat_id, stop, action))
    try:
        await _sleep(REPLY_DELAY)
    finally:
        stop.set()
        await _finish_task(typing_task)

    try:
        try:
            await message.reply_sticker(pick["file_id"], quote=is_group)
        except FloodWait as e:
            wait = _flood_seconds(e)
            if wait > FLOOD_MAX_WAIT:
                raise
            await _sleep(wait + 1)
            await message.reply_sticker(pick["file_id"], quote=is_group)
    except Exception:
        log.warning("sticker send failed (file reference expired?), refreshing list", exc_info=True)
        refresh_stickers_in_background(client)
        return False

    await memory.save_turn(chat_id, message.from_user.id, user_text, pick.get("emoji") or "🙂")
    await log_chat(message, is_group, user_text, f"[sticker {pick.get('emoji') or ''}]".replace(" ]", "]"), "ok", "sticker")
    return True


# ------------------------------------------------------------------ message handling
def schedule_extraction(chat_id: int, user_id: int, text: str) -> None:
    if not LONG_MEMORY or memory is None:
        return
    if memory.should_extract(chat_id, user_id, text):
        spawn(run_extraction(chat_id, user_id))


async def run_extraction(chat_id: int, user_id: int) -> None:
    if not any(p.available() for p in providers.PROVIDERS):   # don't waste a call while everything is down
        return
    history = await memory.load_history(chat_id, user_id)
    await memory.extract_and_store(chat_id, user_id, history, ask_deepseek)


async def process_message(client: Client, message, user, chat, is_group: bool, sticker, text: str, emoji: str) -> None:
    await memory.touch_user(user)

    # sometimes answer a sticker with a sticker
    if sticker and random.random() < STICKER_REPLY_CHANCE:
        if await reply_with_sticker(client, message, sticker, is_group, text):
            return

    history = await memory.load_history(chat.id, user.id)
    style = await build_style_hint(chat.id, user.id, "" if sticker else text)
    facts_block = ""
    if LONG_MEMORY:
        queries = [text] + [m["content"] for m in history if m["role"] == "user"][-2:]
        facts_block = format_facts_block(await memory.relevant_facts(chat.id, user.id, queries))
    if sticker:
        style += (
            f"\n\nNOTE: user ne abhi text nahi, ek sticker bheja hai (emoji: {emoji or '?'}). "
            "Us emoji ke mood pe chhota natural reaction do, jaise dost sticker pe reply karta hai. "
            "Kabhi '[sticker' jaisa text mat likhna."
        )
    messages = (
        [{"role": "system", "content": build_system_prompt(user, is_group, style, facts_block, avoid_block(history))}]
        + history
        + [{"role": "user", "content": text}]
    )

    reply = await generate_reply(client, chat.id, messages, history)
    if not reply:
        fallback_text = await maybe_send_fallback(message, is_group)
        await log_chat(message, is_group, text, fallback_text, "ai_down")
        return

    parts = [p.strip() for p in reply.split("||") if p.strip()][:MAX_BUBBLES] or [reply]
    sent: list[str] = []
    for i, part in enumerate(parts):
        try:
            if i:
                try:
                    await client.send_chat_action(chat.id, ChatAction.TYPING)
                except Exception:
                    pass
                await _sleep(min(1.0 + 0.05 * len(part), 3.0))
            await send_one(message, part, quote=is_group and i == 0)
            sent.append(part)
        except Exception as exc:
            log.warning("send failed in chat %s after %s/%s parts (%s)", chat.id, len(sent), len(parts), type(exc).__name__)
            break
    if not sent:
        await log_chat(message, is_group, text, None, "send_failed")
        return

    # memory always matches what the user actually saw (also after a partial send)
    await memory.save_turn(chat.id, user.id, text, " ".join(sent))
    await log_chat(message, is_group, text, "\n".join(sent), "ok" if len(sent) == len(parts) else "partial",
                   "sticker" if sticker else "text")
    if not sticker:
        schedule_extraction(chat.id, user.id, text)


async def on_message(client: Client, message) -> None:
    user = message.from_user
    if not user or user.is_bot:
        return

    chat = message.chat
    is_group = chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
    sticker = message.sticker
    text = (message.text or "").strip()
    if not sticker and (not text or is_command_text(text)):
        return

    # group: only when tagged / mentioned / replied to
    if is_group and not is_addressed(client, message):
        return
    if not _seen.add_if_new((chat.id, message.id)):          # same update delivered twice
        return
    if not await is_enabled(chat.id):
        return

    emoji = ""
    if sticker:
        emoji = sticker.emoji or ""
        text = f"[sticker: {emoji}]" if emoji else "[sticker]"
    else:
        text = text[:MAX_MESSAGE_CHARS]
        if is_group:
            text = strip_mention(client, text) or "hey"

    key = (chat.id, user.id)
    if _pending.get(key, 0) >= MAX_PENDING_PER_USER:         # flood protection: drop, don't queue forever
        log.warning("dropping message from user %s in chat %s: too many pending", user.id, chat.id)
        return
    _pending[key] = _pending.get(key, 0) + 1
    try:
        async with _locks.get(key):
            await process_message(client, message, user, chat, is_group, sticker, text, emoji)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("unhandled error while processing a message in chat %s", chat.id)
    finally:
        left = _pending.get(key, 1) - 1
        if left <= 0:
            _pending.pop(key, None)
        else:
            _pending[key] = left


# ------------------------------------------------------------------ lifecycle
async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def shutdown(*, client=None, session=None, mongo=None, runner=None, health=None,
                   timeout: float = SHUTDOWN_TIMEOUT) -> None:
    """Close everything in order; a failure in one step never skips the others."""
    if health is not None:
        health.stopping = True

    tasks = [t for t in _bg_tasks if not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)

    async def step(name: str, fn) -> None:
        if fn is None:
            return
        try:
            await asyncio.wait_for(_maybe_await(fn()), timeout)
        except Exception as exc:
            log.warning("shutdown: %s did not close cleanly (%s)", name, type(exc).__name__)

    await step("telegram client", getattr(client, "stop", None))
    await step("http session", getattr(session, "close", None))
    await step("mongodb client", getattr(mongo, "close", None))
    await step("health server", getattr(runner, "cleanup", None))
    log.info("shutdown complete")


async def _retry_indexes() -> None:
    for _ in range(10):
        await _sleep(30)
        if await memory.ensure_indexes():
            log.info("MongoDB indexes created")
            return


async def main() -> None:
    global memory
    setup_logging(LOG_LEVEL, SECRETS)
    runner = mongo = session = client = None
    try:
        runner = await start_health_server(                    # first, so Render sees the open port
            state, PORT, setup=lambda app: dashboard.setup(app, lambda: memory)
        )
        log.info("Health server listening on port %s", PORT)
        log.info("Owner dashboard: %s", "enabled at /admin" if dashboard.ADMIN_PASSWORD else "disabled (set ADMIN_PASSWORD to enable)")

        mongo = AsyncIOMotorClient(
            MONGO_URI, maxPoolSize=DB_MAX_POOL, serverSelectionTimeoutMS=DB_TIMEOUT_MS, connectTimeoutMS=DB_TIMEOUT_MS
        )
        memory = Memory(mongo[DB_NAME])
        state.ping = memory.ping
        if not await memory.ensure_indexes():
            log.warning("MongoDB not ready yet - bot starts in degraded mode and keeps retrying")
            spawn(_retry_indexes())

        session = aiohttp.ClientSession()
        providers.set_http(session)

        client = Client(
            "girl_chatbot",
            api_id=API_ID,
            api_hash=API_HASH,
            session_string=STRING_SESSION,
            in_memory=True,
        )
        # order matters: the first matching handler of a group wins, so commands come before on_message
        client.add_handler(
            MessageHandler(
                on_command,
                filters.me & filters.command(["chatoff", "chaton", "stickers"], prefixes=CMD_PREFIX),
            )
        )
        client.add_handler(
            MessageHandler(
                on_user_command,
                (filters.private | filters.group)
                & filters.incoming
                & ~filters.me
                & ~filters.bot
                & filters.command(list(USER_COMMANDS), prefixes=CMD_PREFIX),
            )
        )
        client.add_handler(
            MessageHandler(
                on_message,
                (filters.private | filters.group)
                & filters.incoming
                & (filters.text | filters.sticker)
                & ~filters.me
                & ~filters.bot
                & ~filters.service,
            )
        )

        await client.start()
        state.client = client
        state.ready = True
        log.info("Started as %s (@%s)", client.me.first_name, client.me.username)
        refresh_stickers_in_background(client)  # load the sticker packs added to this account
        await idle()                            # returns on SIGTERM / SIGINT (Render stop, Ctrl+C)
    finally:
        await shutdown(client=client, session=session, mongo=mongo, runner=runner, health=state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
