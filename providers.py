"""
AI provider layer: primary provider (DeepSeek or any OpenAI-compatible API) + optional backups.

Error handling (matches DeepSeek's documented codes 400/401/402/422/429/500/503):
- 401/403 (bad key), 402 (no balance), 404 (wrong model/URL): long cooldown, fail over at once, never retried
- 400/422 (bad request/params): if an optional parameter was rejected, retry once without it;
  otherwise short cooldown + fail over (retrying the same request would not help)
- 429: honour Retry-After, cooldown, fail over at once (no hammering)
- 500/502/503/504, timeouts, connection errors: bounded retries with exponential backoff + jitter,
  then cooldown + fail over
- empty / filtered / malformed replies are handled without crashing
Secrets are never logged (request headers/payloads are not printed).
"""
import asyncio
import logging
import random
import re
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import aiohttp

from config import (
    API_CONCURRENCY,
    BACKOFF_BASE,
    BACKOFF_MAX,
    CONNECT_TIMEOUT,
    COOLDOWN_BAD_REQUEST,
    COOLDOWN_DAILY,
    COOLDOWN_ERROR,
    COOLDOWN_RATE,
    FREQUENCY_PENALTY,
    MAX_REPLY_CHARS,
    MAX_TOKENS,
    PERSONA_NAME,
    PROVIDER_CONFIGS,
    PROVIDER_RETRIES,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    TOP_P,
)

log = logging.getLogger("girl-chatbot.providers")

http: "aiohttp.ClientSession | None" = None     # set by main via set_http()
_sleep = asyncio.sleep                           # indirection so tests can skip real waiting


def set_http(session) -> None:
    global http
    http = session


# per-event-loop semaphore (safe across test loops, bounded API concurrency in production)
_sem = None
_sem_loop = None


def _api_sem() -> asyncio.Semaphore:
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem, _sem_loop = asyncio.Semaphore(API_CONCURRENCY), loop
    return _sem


def clean_reply(text: str) -> str:
    """Remove artefacts the model sometimes adds: 'Riya:' prefix, wrapping quotes, '[sticker: ..]' tags."""
    text = text.strip()
    for _ in range(3):  # peel layers in any order, e.g.  Riya: "[sticker: x] hi"
        before = text
        text = re.sub(rf"^{re.escape(PERSONA_NAME)}\s*:\s*", "", text, flags=re.I)
        text = text.strip().strip('"“”').strip()
        text = re.sub(r"^\[sticker[^\]]*\]\s*", "", text, flags=re.I)
        if text == before:
            break
    return text[:MAX_REPLY_CHARS].strip()


class Provider:
    """One OpenAI-compatible chat API (URL + key + model)."""

    def __init__(self, name: str, url: str, key: str, model: str, extra: dict | None = None):
        self.name, self.url, self.key, self.model = name, url, key, model
        self.user_extra = dict(extra or {})     # explicit JSON from *_EXTRA env: always sent
        self.cooldown_until = 0.0               # monotonic time until which this provider is skipped
        self.strip_extras = False               # provider rejected our optional params -> stop sending them
        host = (urlparse(url).hostname or "").lower()
        if host.endswith("deepseek.com"):
            self.kind = "deepseek"
        elif host.endswith("groq.com"):
            self.kind = "groq"
        else:
            self.kind = "generic"

    # ---- availability
    def available(self) -> bool:
        return self.cooldown_until <= time.monotonic()

    def cool_down(self, seconds: float) -> None:
        self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)

    # ---- request building
    def auto_extras(self) -> dict:
        """Optional provider-specific params; dropped automatically if the provider rejects them."""
        extras: dict = {}
        if self.kind == "deepseek":
            # official docs: thinking mode is ON by default (slow, burns tokens, content can come back empty
            # with a small max_tokens). A chat bot wants plain fast replies. frequency/presence_penalty are
            # deprecated (no effect) and top_p is ignored outside thinking mode, so none of those are sent.
            extras["thinking"] = {"type": "disabled"}
        else:
            extras["top_p"] = TOP_P
            if FREQUENCY_PENALTY:
                extras["frequency_penalty"] = FREQUENCY_PENALTY
            if self.kind == "groq" and self.model.startswith("openai/gpt-oss"):
                extras["reasoning_effort"] = "low"      # fewer hidden thinking tokens (unverified on live API)
        return extras

    def build_payload(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if not self.strip_extras:
            payload.update(self.auto_extras())
        payload.update(self.user_extra)
        return payload


PROVIDERS = [Provider(**cfg) for cfg in PROVIDER_CONFIGS]


# ------------------------------------------------------------------ helpers
def parse_retry_after(value) -> "float | None":
    if not value:
        return None
    try:
        return min(max(float(value), 1.0), 3600.0)
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        delta = parsedate_to_datetime(str(value)).timestamp() - time.time()
        return min(max(delta, 1.0), 3600.0)
    except Exception:
        return None


def _is_daily_limit(body_low: str) -> bool:
    return any(k in body_low for k in ("per day", "daily", "free_trial", "free-trial", "tokens per day", "quota"))


def _mentions_optional_param(body_low: str, params) -> bool:
    if any(p.lower() in body_low for p in params):
        return True
    return any(k in body_low for k in ("unsupported", "not supported", "unrecognized", "unknown param",
                                       "extra inputs", "invalid parameter", "unexpected"))


async def _backoff(attempt: int, retry_after: "float | None" = None) -> None:
    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) * random.uniform(0.7, 1.3)
    if retry_after:
        delay = max(delay, min(retry_after, BACKOFF_MAX * 2))
    await _sleep(delay)


async def _post(p: Provider, payload: dict):
    """One HTTP request. Returns (status, body_text, headers, json_or_None)."""
    headers = {"Authorization": f"Bearer {p.key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with _api_sem():
        async with http.post(p.url, json=payload, headers=headers, timeout=timeout) as r:
            if r.status == 200:
                try:
                    data = await r.json(content_type=None)
                except (ValueError, aiohttp.ClientError):
                    data = None
                return 200, "", r.headers, data
            return r.status, (await r.text())[:300], r.headers, None


def _parse_choice(data):
    """Returns (content, finish_reason) or (None, 'malformed')."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, "malformed"
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content, finish


# ------------------------------------------------------------------ one provider
async def _call(p: Provider, messages: list[dict], max_tokens: int, temperature: "float | None" = None):
    """One logical request to one provider. Returns (text, outcome): ok | empty | filtered | error."""
    temperature = TEMPERATURE if temperature is None else temperature
    attempt = 0
    while True:
        attempt += 1
        payload = p.build_payload(messages, max_tokens, temperature)
        try:
            status, body, headers, data = await _post(p, payload)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
            if attempt <= PROVIDER_RETRIES:
                log.warning("[%s] %s, retry %s/%s", p.name, type(exc).__name__, attempt, PROVIDER_RETRIES)
                await _backoff(attempt)
                continue
            p.cool_down(COOLDOWN_ERROR)
            log.error("[%s] unreachable (%s) after %s attempts, skipping for %ss",
                      p.name, type(exc).__name__, attempt, COOLDOWN_ERROR)
            return None, "error"
        except Exception as exc:  # defensive: never let an odd client error crash the handler
            p.cool_down(COOLDOWN_ERROR)
            log.error("[%s] unexpected %s while calling API", p.name, type(exc).__name__)
            return None, "error"

        retry_after = parse_retry_after(headers.get("Retry-After") if headers else None)
        low = body.lower()

        if status == 200:
            content, finish = _parse_choice(data)
            if finish == "malformed":
                p.cool_down(COOLDOWN_ERROR)
                log.error("[%s] malformed API response, skipping for %ss", p.name, COOLDOWN_ERROR)
                return None, "error"
            if finish in ("insufficient_system_resource", "aborted") and attempt <= PROVIDER_RETRIES:
                log.warning("[%s] finish_reason=%s, retry %s", p.name, finish, attempt)
                await _backoff(attempt)
                continue
            text = clean_reply(content) if isinstance(content, str) and content.strip() else ""
            if not text:
                log.warning("[%s] empty reply (finish_reason=%s)", p.name, finish)
                return None, "filtered" if finish == "content_filter" else "empty"
            return text, "ok"

        if status in (400, 422):
            optional = [] if p.strip_extras else list(p.auto_extras())
            if optional and _mentions_optional_param(low, optional):
                p.strip_extras = True
                log.warning("[%s] HTTP %s: optional parameters rejected, retrying without %s",
                            p.name, status, ",".join(optional))
                continue
            p.cool_down(COOLDOWN_BAD_REQUEST)
            log.error("[%s] HTTP %s invalid request, failing over: %s", p.name, status, body[:200])
            return None, "error"

        if status in (401, 403):
            p.cool_down(COOLDOWN_DAILY)
            log.error("[%s] HTTP %s authentication failed - check the API key", p.name, status)
            return None, "error"
        if status == 402:
            p.cool_down(COOLDOWN_DAILY)
            log.error("[%s] HTTP 402 insufficient balance - top up the account", p.name)
            return None, "error"
        if status == 404:
            p.cool_down(COOLDOWN_DAILY)
            log.error("[%s] HTTP 404 model or URL not found: %s", p.name, body[:200])
            return None, "error"
        if status == 429:
            wait = retry_after or (COOLDOWN_DAILY if _is_daily_limit(low) else COOLDOWN_RATE)
            p.cool_down(wait)
            log.error("[%s] HTTP 429 rate limited, skipping for %ss: %s", p.name, int(wait), body[:160])
            return None, "error"
        if status in (500, 502, 503, 504):
            if attempt <= PROVIDER_RETRIES:
                log.warning("[%s] HTTP %s, retry %s/%s", p.name, status, attempt, PROVIDER_RETRIES)
                await _backoff(attempt, retry_after)
                continue
            p.cool_down(retry_after or COOLDOWN_ERROR)
            log.error("[%s] HTTP %s after %s attempts, skipping this provider", p.name, status, attempt)
            return None, "error"

        p.cool_down(COOLDOWN_ERROR)
        log.error("[%s] unexpected HTTP %s: %s", p.name, status, body[:160])
        return None, "error"


# ------------------------------------------------------------------ the chain
async def ask_deepseek(messages: list[dict], max_tokens: "int | None" = None,
                       temperature: "float | None" = None) -> "str | None":
    """Try providers in order (primary first). Returns None when every provider failed/unavailable.
    Empty reply -> one retry with 3x tokens (reasoning models can burn the budget on thinking)."""
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    ready = [p for p in PROVIDERS if p.available()]
    if not ready:
        log.error("All API providers are cooling down - no AI reply possible right now")
        return None
    for p in ready:
        text, outcome = await _call(p, messages, max_tokens, temperature)
        if outcome == "empty":
            text, outcome = await _call(p, messages, max_tokens * 3, temperature)
        if text:
            if p is not PROVIDERS[0]:
                log.info("Reply came from backup provider [%s]", p.name)
            return text
    return None


def providers_status() -> list[dict]:
    """For the health endpoint (names and timings only, never keys)."""
    t = time.monotonic()
    return [
        {"name": p.name, "ready": p.cooldown_until <= t, "cooldown_s": max(0, int(p.cooldown_until - t))}
        for p in PROVIDERS
    ]
