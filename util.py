"""Small shared helpers: log redaction + bounded (non-leaking) containers."""
import logging
import re
from collections import OrderedDict

# generic key shapes (sk-..., gsk_..., Bearer xxx) are redacted even if not in the secrets list
_KEY_PATTERNS = [
    re.compile(r"\b(?:sk|gsk|nvapi|xai)[-_][A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{12,}"),
    re.compile(r"(?i)(mongodb(?:\+srv)?://[^:/\s]+:)[^@\s]+(@)"),
]


class RedactingFormatter(logging.Formatter):
    """Replaces known secrets (API keys, session string, DB password) in every log line,
    including tracebacks, so they never reach Render's log viewer."""

    def __init__(self, fmt: str, secrets: list[str] | None = None):
        super().__init__(fmt)
        self._secrets: list[str] = []
        self.set_secrets(secrets or [])

    def set_secrets(self, secrets: list[str]) -> None:
        # longest first so a key is not partially masked by a shorter one
        self._secrets = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***")
        for pattern in _KEY_PATTERNS:
            if pattern.groups:
                text = pattern.sub(lambda m: m.group(1) + "***" + (m.group(2) if m.re.groups > 1 else ""), text)
            else:
                text = pattern.sub("***", text)
        return text

    def format(self, record: logging.LogRecord) -> str:
        return self.redact(super().format(record))


def setup_logging(level: str, secrets: list[str]) -> RedactingFormatter:
    fmt = RedactingFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", secrets)
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    return fmt


class BoundedDict(OrderedDict):
    """Dict that forgets its oldest entries beyond max_size (no unbounded growth)."""

    def __init__(self, max_size: int = 2000):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_size:
            self.popitem(last=False)


class BoundedSet:
    """Remembers the last `max_size` keys; add_if_new() tells whether a key was seen before."""

    def __init__(self, max_size: int = 2000):
        self._d = BoundedDict(max_size)

    def add_if_new(self, key) -> bool:
        if key in self._d:
            return False
        self._d[key] = True
        return True

    def __len__(self) -> int:
        return len(self._d)


class LockPool:
    """asyncio locks keyed by (chat, user); old *unlocked* locks are evicted so memory stays bounded."""

    def __init__(self, max_size: int = 2000):
        import asyncio

        self._asyncio = asyncio
        self._locks: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        lock = self._locks.get(key)
        if lock is None:
            lock = self._asyncio.Lock()
            self._locks[key] = lock
        self._locks.move_to_end(key)
        if len(self._locks) > self.max_size:
            for k in list(self._locks):
                if len(self._locks) <= self.max_size:
                    break
                if k != key and not self._locks[k].locked():
                    del self._locks[k]
        return lock

    def __len__(self) -> int:
        return len(self._locks)
