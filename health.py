"""
Health endpoint logic.

GET / and GET /health return JSON:
  status: "starting" (still booting) | "healthy" | "degraded" (running, but Telegram/MongoDB/AI is down)
          | "stopping" (shutdown in progress)
HTTP is always 200 so pingers (and Render's health check) keep the service alive and a MongoDB blip does
not make Render restart the bot. Add ?strict=1 to get 503 unless the status is "healthy".
"""
import asyncio
import time

from config import HEALTH_MONGO_CACHE, HEALTH_PING_TIMEOUT
import providers


class HealthState:
    def __init__(self, ping=None, providers_status=None):
        self.started = time.monotonic()
        self.ready = False          # set once the Telegram client is running
        self.stopping = False       # set when shutdown begins
        self.client = None          # Telegram client (for is_connected)
        self.ping = ping            # async callable that raises if MongoDB is unreachable
        self.providers_status = providers_status or providers.providers_status
        self._mongo = (0.0, None)   # (checked_at, ok) cache

    async def mongo_ok(self) -> "bool | None":
        if self.ping is None:
            return None
        checked_at, ok = self._mongo
        if ok is not None and time.monotonic() - checked_at < HEALTH_MONGO_CACHE:
            return ok
        try:
            await asyncio.wait_for(self.ping(), HEALTH_PING_TIMEOUT)
            ok = True
        except Exception:
            ok = False
        self._mongo = (time.monotonic(), ok)
        return ok


async def check_health(state: HealthState) -> dict:
    uptime = int(time.monotonic() - state.started)
    if state.stopping:
        return {"status": "stopping", "uptime_s": uptime}
    if not state.ready:
        return {"status": "starting", "uptime_s": uptime}

    telegram_ok = bool(getattr(state.client, "is_connected", False))
    mongo = await state.mongo_ok()
    prov = state.providers_status()
    ready_providers = sum(1 for p in prov if p["ready"])

    healthy = telegram_ok and mongo is not False and ready_providers > 0
    return {
        "status": "healthy" if healthy else "degraded",
        "uptime_s": uptime,
        "checks": {
            "telegram": telegram_ok,
            "mongo": mongo,
            "providers_ready": ready_providers,
            "providers_total": len(prov),
        },
        "providers": prov,
    }


def make_app(state: HealthState, setup=None):
    from aiohttp import web

    async def handler(request):
        data = await check_health(state)
        strict = request.query.get("strict", "").lower() in ("1", "true", "yes")
        status = 503 if (strict and data["status"] != "healthy") else 200
        return web.json_response(data, status=status)

    app = web.Application()
    app.router.add_get("/", handler)
    app.router.add_get("/health", handler)
    if setup is not None:
        setup(app)                  # e.g. the owner dashboard routes
    return app


async def start_health_server(state: HealthState, port: int, setup=None):
    from aiohttp import web

    runner = web.AppRunner(make_app(state, setup))
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    return runner
