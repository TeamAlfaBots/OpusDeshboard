"""Fake pyrogram / motor / aiohttp modules (installed into sys.modules before the bot is imported)."""
import asyncio
import sys
import types
from types import SimpleNamespace

_idle_event = None


def reset_idle():
    global _idle_event
    _idle_event = asyncio.Event()
    return _idle_event


def install():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    # ---------------- pyrogram
    class _F:
        def __and__(self, o): return self
        def __or__(self, o): return self
        def __invert__(self): return self
        def __call__(self, *a, **k): return self

    class _Filters:
        def __getattr__(self, name): return _F()

    class Req:
        def __init__(self, name, **kw):
            self.name = name
            self.__dict__.update(kw)

    class DocumentAttributeSticker:
        def __init__(self, alt): self.alt = alt

    raw = SimpleNamespace(
        functions=SimpleNamespace(messages=SimpleNamespace(
            GetAllStickers=lambda hash: Req("GetAllStickers"),
            GetStickerSet=lambda stickerset, hash: Req("GetStickerSet", stickerset=stickerset))),
        types=SimpleNamespace(
            InputStickerSetID=lambda id, access_hash: SimpleNamespace(id=id, access_hash=access_hash),
            DocumentAttributeSticker=DocumentAttributeSticker),
    )

    class FileType:
        STICKER = "sticker"

    class FileId:
        def __init__(self, file_type=None, dc_id=None, media_id=None, access_hash=None, file_reference=None):
            self.media_id = media_id
        def encode(self): return f"enc_{self.media_id}"
        @staticmethod
        def decode(fid): return SimpleNamespace(media_id=int(str(fid).split("_")[-1]))

    class FloodWait(Exception):
        def __init__(self, value=0):
            super().__init__(f"FloodWait {value}")
            self.value = value

    class ChatAction:
        TYPING = "typing"
        CHOOSE_STICKER = "choose_sticker"

    class ChatType:
        PRIVATE = "private"
        GROUP = "group"
        SUPERGROUP = "supergroup"

    class MessageHandler:
        def __init__(self, callback, flt=None):
            self.callback, self.filters = callback, flt

    class Client:
        instances = []

        def __init__(self, name, **kw):
            self.name, self.kw, self.handlers = name, kw, []
            self.me = None
            self.started = self.stopped = False
            Client.instances.append(self)

        def add_handler(self, h): self.handlers.append(h)
        async def start(self):
            self.started = True
            self.me = SimpleNamespace(id=1, first_name="Cherry", username="cherry")
        async def stop(self): self.stopped = True
        @property
        def is_connected(self): return self.started and not self.stopped
        async def invoke(self, req): return SimpleNamespace(sets=[], documents=[])
        async def send_chat_action(self, *a): pass

    async def idle():
        await _idle_event.wait()

    mod("pyrogram", Client=Client, filters=_Filters(), idle=idle, raw=raw)
    mod("pyrogram.enums", ChatAction=ChatAction, ChatType=ChatType)
    mod("pyrogram.errors", FloodWait=FloodWait)
    mod("pyrogram.file_id", FileId=FileId, FileType=FileType)
    mod("pyrogram.handlers", MessageHandler=MessageHandler)

    # ---------------- motor
    class AsyncIOMotorClient:
        instances = []

        def __init__(self, uri, **kw):
            from tests.fakes import FakeDB
            self.uri, self.kw, self.closed = uri, kw, False
            self._db = FakeDB()
            AsyncIOMotorClient.instances.append(self)

        def __getitem__(self, name): return self._db
        def close(self): self.closed = True

    mod("motor"); mod("motor.motor_asyncio", AsyncIOMotorClient=AsyncIOMotorClient)

    # ---------------- aiohttp
    class ClientError(Exception): pass

    class ClientTimeout:
        def __init__(self, total=None, connect=None): self.total, self.connect = total, connect

    class ClientSession:
        def __init__(self, *a, **k): self.closed = False
        async def close(self): self.closed = True

    class _Router:
        def __init__(self): self.routes, self.post_routes = {}, {}
        def add_get(self, path, handler): self.routes[path] = handler
        def add_post(self, path, handler): self.post_routes[path] = handler

    class Application:
        def __init__(self): self.router = _Router()

    class AppRunner:
        instances = []
        def __init__(self, app): self.app, self.cleaned = app, False; AppRunner.instances.append(self)
        async def setup(self): pass
        async def cleanup(self): self.cleaned = True

    class TCPSite:
        def __init__(self, runner, host, port): self.runner, self.host, self.port = runner, host, port
        async def start(self): pass

    class Response:
        def __init__(self, text=None, status=200, content_type=None, headers=None):
            self.text, self.status, self.content_type = text, status, content_type
            self.headers, self.cookies, self.deleted_cookies = dict(headers or {}), {}, []
        def set_cookie(self, name, value, **kw): self.cookies[name] = {"value": value, **kw}
        def del_cookie(self, name, **kw): self.deleted_cookies.append(name)

    def json_response(data, status=200, headers=None):
        r = Response(text=None, status=status, headers=headers)
        r.data = data
        return r

    web = mod("aiohttp.web", Application=Application, AppRunner=AppRunner, TCPSite=TCPSite,
              json_response=json_response, Response=Response)
    mod("aiohttp", ClientError=ClientError, ClientTimeout=ClientTimeout, ClientSession=ClientSession, web=web)
