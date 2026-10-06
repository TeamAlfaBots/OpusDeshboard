"""In-memory fakes: MongoDB, HTTP API, Telegram client/messages."""
import asyncio
import itertools
from types import SimpleNamespace

from pyrogram.enums import ChatType


# ------------------------------------------------------------------ MongoDB
class DuplicateKeyError(Exception):
    code = 11000


class Cursor:
    def __init__(self, docs): self.docs = list(docs)

    def sort(self, key, direction=1):
        def k(d):
            v = d.get(key)
            return (v is None, v if v is not None else 0)
        self.docs.sort(key=k, reverse=direction == -1)
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    async def to_list(self, length=None):
        return [dict(d) for d in (self.docs if length is None else self.docs[:length])]


class Collection:
    _ids = itertools.count(1)

    def __init__(self, db, name):
        self.db, self.name = db, name
        self.docs, self.indexes, self.unique = [], [], []

    def _check(self):
        if self.db.fail:
            raise ConnectionError("mongo down")

    @staticmethod
    def _match(doc, flt):
        import re as _re
        for key, want in (flt or {}).items():
            if key == "$or":
                if not any(Collection._match(doc, sub) for sub in want):
                    return False
                continue
            have = doc.get(key)
            if isinstance(want, dict) and any(k.startswith("$") for k in want):
                for op, val in want.items():
                    if op == "$regex":
                        flags = _re.I if "i" in want.get("$options", "") else 0
                        if not isinstance(have, str) or not _re.search(val, have, flags):
                            return False
                    elif op == "$lt":
                        if have is None or not have < val:
                            return False
                    elif op == "$ne":
                        if have == val:
                            return False
                continue
            if isinstance(have, list) and not isinstance(want, list):
                if want not in have:
                    return False
            elif have != want:
                return False
        return True

    def find(self, flt=None, projection=None):
        self._check()
        return Cursor(d for d in self.docs if self._match(d, flt))

    async def find_one(self, flt):
        self._check()
        return next((dict(d) for d in self.docs if self._match(d, flt)), None)

    def _unique_ok(self, doc):
        for keys in self.unique:
            for d in self.docs:
                if all(d.get(k) == doc.get(k) for k in keys):
                    raise DuplicateKeyError("E11000 duplicate key")

    async def insert_one(self, doc):
        self._check()
        doc = dict(doc)
        self._unique_ok(doc)
        doc.setdefault("_id", next(self._ids))
        self.docs.append(doc)

    async def insert_many(self, docs, ordered=True):
        self._check()
        for d in docs:
            await self.insert_one(d)

    async def update_one(self, flt, update, upsert=False):
        self._check()
        for d in self.docs:
            if self._match(d, flt):
                self._apply(d, update)
                return
        if upsert:
            doc = dict(flt)
            doc.update(update.get("$setOnInsert", {}))
            self._apply(doc, update)
            doc.setdefault("_id", next(self._ids))
            self.docs.append(doc)

    @staticmethod
    def _apply(doc, update):
        doc.update(update.get("$set", {}))
        for k, v in update.get("$inc", {}).items():
            doc[k] = doc.get(k, 0) + v
        for k, v in update.get("$addToSet", {}).items():
            items = v["$each"] if isinstance(v, dict) and "$each" in v else [v]
            current = doc.setdefault(k, [])
            for item in items:
                if item not in current:
                    current.append(item)

    async def delete_one(self, flt):
        self._check()
        for i, d in enumerate(self.docs):
            if self._match(d, flt):
                del self.docs[i]
                return

    async def delete_many(self, flt):
        self._check()
        self.docs[:] = [d for d in self.docs if not self._match(d, flt)]

    async def count_documents(self, flt):
        self._check()
        return sum(1 for d in self.docs if self._match(d, flt))

    async def create_index(self, keys, **kw):
        self._check()
        if self.db.ttl_conflict and kw.get("expireAfterSeconds") is not None:
            self.db.ttl_conflict = False
            raise RuntimeError("IndexOptionsConflict")
        spec = keys if isinstance(keys, list) else [(keys, 1)]
        self.indexes.append((spec, kw))
        if kw.get("unique"):
            self.unique.append([k for k, _ in spec])


class FakeDB:
    def __init__(self):
        self.fail = False
        self.ttl_conflict = False
        self.commands = []
        self._cols = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._cols.setdefault(name, Collection(self, name))

    async def command(self, *args, **kw):
        if self.fail:
            raise ConnectionError("mongo down")
        self.commands.append((args, kw))


# ------------------------------------------------------------------ HTTP API
class FakeResponse:
    def __init__(self, status=200, data=None, text="", headers=None, bad_json=False):
        self.status, self._data, self._text = status, data, text
        self.headers, self._bad_json = headers or {}, bad_json

    async def json(self, content_type=None):
        if self._bad_json:
            raise ValueError("not json")
        return self._data

    async def text(self): return self._text
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


def ok(text, finish="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": text}, "finish_reason": finish}]})


def err(status, text="", headers=None):
    return FakeResponse(status, text=text, headers=headers)


class FakeHTTP:
    def __init__(self):
        self.script, self.requests = {}, []

    def queue(self, url, *items):
        self.script.setdefault(url, []).extend(items)

    def calls(self, url):
        return [r for r in self.requests if r["url"] == url]

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        q = self.script.get(url)
        item = q.pop(0) if q else ok("default reply")
        if isinstance(item, BaseException):
            raise item
        return item


# ------------------------------------------------------------------ Telegram
class FakeClient:
    def __init__(self, username="cherry"):
        self.me = SimpleNamespace(id=1, username=username, first_name="Cherry")
        self.actions, self.sticker_sets, self.invoked = [], {}, []
        self.is_connected = True

    async def send_chat_action(self, chat_id, action): self.actions.append((chat_id, action))

    async def invoke(self, req):
        self.invoked.append(req.name)
        if req.name == "GetAllStickers":
            return SimpleNamespace(sets=[SimpleNamespace(id=i, access_hash=1, short_name=f"p{i}") for i in self.sticker_sets])
        return SimpleNamespace(documents=self.sticker_sets[req.stickerset.id])


_msg_ids = itertools.count(100)


def user(uid, name="Aman", is_bot=False):
    return SimpleNamespace(id=uid, first_name=name, username=f"user{uid}", is_bot=is_bot)


class FakeMessage:
    def __init__(self, text=None, *, uid=42, chat_id=None, group=False, sticker=None, mentioned=False,
                 reply_to=None, entities=None, command=None, name="Aman"):
        self.id = next(_msg_ids)
        self.text, self.sticker, self.mentioned = text, sticker, mentioned
        self.reply_to_message, self.entities, self.command = reply_to, entities, command
        self.from_user = user(uid, name)
        self.chat = SimpleNamespace(id=chat_id if chat_id is not None else (-1001 if group else uid),
                                    type=ChatType.SUPERGROUP if group else ChatType.PRIVATE)
        self.sent, self.deleted, self.edits = [], False, []
        self.fail_on_send = None     # callable(call_no) -> exception or None
        self._send_calls = 0

    async def reply_text(self, text, quote=None):
        self._send_calls += 1
        if self.fail_on_send:
            exc = self.fail_on_send(self._send_calls)
            if exc:
                raise exc
        self.sent.append(("text", text, bool(quote)))

    async def reply_sticker(self, file_id, quote=None):
        self._send_calls += 1
        if self.fail_on_send:
            exc = self.fail_on_send(self._send_calls)
            if exc:
                raise exc
        self.sent.append(("sticker", file_id, bool(quote)))

    async def edit_text(self, text): self.edits.append(text)
    async def delete(self): self.deleted = True

    @property
    def texts(self): return [s[1] for s in self.sent if s[0] == "text"]


def make_sticker(media_id, emoji):
    return SimpleNamespace(file_unique_id=f"u{media_id}", file_id=f"enc_{media_id}", emoji=emoji, set_name=None)


def make_doc(doc_id, emoji):
    from pyrogram import raw
    return SimpleNamespace(id=doc_id, dc_id=2, access_hash=7, file_reference=b"r",
                           attributes=[raw.types.DocumentAttributeSticker(emoji)])


async def drain(tasks):
    """Wait for background tasks (memory extraction etc.)."""
    tasks = [t for t in tasks if not t.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
