"""Health endpoint, graceful shutdown (full main() lifecycle), config validation, log redaction, bounded containers."""
import asyncio
import logging
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import health
import main
import providers
from health import HealthState, check_health, make_app
from tests import stubs
from tests.fakes import FakeClient, FakeDB
from util import BoundedDict, BoundedSet, LockPool, RedactingFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prov(ready=True):
    return lambda: [{"name": "primary", "ready": ready, "cooldown_s": 0 if ready else 99}]


class TestHealth(unittest.IsolatedAsyncioTestCase):
    async def test_starting_then_healthy(self):
        calls = []

        async def ping(): calls.append(1)
        st = HealthState(ping=ping, providers_status=prov())
        self.assertEqual((await check_health(st))["status"], "starting")
        st.ready, st.client = True, FakeClient()
        data = await check_health(st)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["checks"], {"telegram": True, "mongo": True, "providers_ready": 1, "providers_total": 1})

    async def test_degraded_when_mongo_down_telegram_down_or_no_provider(self):
        async def bad_ping(): raise ConnectionError("down")

        st = HealthState(ping=bad_ping, providers_status=prov())
        st.ready, st.client = True, FakeClient()
        self.assertEqual((await check_health(st))["status"], "degraded")
        self.assertIs((await check_health(st))["checks"]["mongo"], False)

        async def good_ping(): return None
        st = HealthState(ping=good_ping, providers_status=prov())
        st.ready, st.client = True, SimpleNamespace(is_connected=False)
        self.assertEqual((await check_health(st))["status"], "degraded")

        st = HealthState(ping=good_ping, providers_status=prov(ready=False))
        st.ready, st.client = True, FakeClient()
        self.assertEqual((await check_health(st))["status"], "degraded")

    async def test_slow_mongo_ping_times_out_as_degraded(self):
        async def slow(): await asyncio.sleep(5)
        st = HealthState(ping=slow, providers_status=prov())
        st.ready, st.client = True, FakeClient()
        with mock.patch.object(health, "HEALTH_PING_TIMEOUT", 0.05):
            self.assertEqual((await check_health(st))["status"], "degraded")

    async def test_mongo_ping_is_cached_between_polls(self):
        calls = []

        async def ping(): calls.append(1)
        st = HealthState(ping=ping, providers_status=prov())
        st.ready, st.client = True, FakeClient()
        for _ in range(5):
            await check_health(st)
        self.assertEqual(len(calls), 1)                          # pingers polling every minute don't hammer MongoDB

    async def test_stopping(self):
        st = HealthState(providers_status=prov())
        st.ready = st.stopping = True
        self.assertEqual((await check_health(st))["status"], "stopping")

    async def test_http_handler_both_routes_and_strict_mode(self):
        async def bad(): raise ConnectionError("x")
        st = HealthState(ping=bad, providers_status=prov())
        st.ready, st.client = True, FakeClient()
        app = make_app(st)
        self.assertEqual(set(app.router.routes), {"/", "/health"})            # both endpoints preserved
        normal = await app.router.routes["/health"](SimpleNamespace(query={}))
        self.assertEqual((normal.status, normal.data["status"]), (200, "degraded"))   # 200: Render must not restart on a DB blip
        strict = await app.router.routes["/"](SimpleNamespace(query={"strict": "1"}))
        self.assertEqual(strict.status, 503)
        st2 = HealthState(providers_status=prov())
        st2.ready, st2.client = True, FakeClient()
        ok = await make_app(st2).router.routes["/health"](SimpleNamespace(query={"strict": "1"}))
        self.assertEqual((ok.status, ok.data["status"]), (200, "healthy"))

    async def test_health_payload_never_contains_secrets(self):
        st = HealthState(providers_status=providers.providers_status)
        st.ready, st.client = True, FakeClient()
        text = str(await check_health(st))
        for secret in main.SECRETS:
            self.assertNotIn(secret, text)


class TestShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_closes_everything_in_order_even_if_one_step_fails(self):
        order = []

        class Obj:
            def __init__(self, name, fn, fail=False):
                self._n, self._fail = name, fail
                setattr(self, fn, self._go)

            async def _go(self):
                order.append(self._n)
                if self._fail:
                    raise RuntimeError("close failed")

        class SyncClose:
            def close(self): order.append("mongo")

        bg_started = asyncio.Event()

        async def background():
            bg_started.set()
            await asyncio.sleep(60)
        task = main.spawn(background())
        await bg_started.wait()
        st = HealthState()
        await main.shutdown(client=Obj("client", "stop", fail=True), session=Obj("session", "close"),
                            mongo=SyncClose(), runner=Obj("runner", "cleanup"), health=st)
        self.assertEqual(order, ["client", "session", "mongo", "runner"])      # one failure did not skip the rest
        self.assertTrue(task.cancelled())                                      # background tasks cancelled
        self.assertTrue(st.stopping)
        await main.shutdown()                                                  # safe with nothing to close

    async def test_full_lifecycle_start_serve_and_graceful_stop(self):
        stubs.Client = sys.modules["pyrogram"].Client
        stubs.Client.instances.clear()
        sys.modules["aiohttp.web"].AppRunner.instances.clear()
        sys.modules["motor.motor_asyncio"].AsyncIOMotorClient.instances.clear()
        event = stubs.reset_idle()
        main.state = HealthState()
        main.state.providers_status = prov()

        runner_task = asyncio.create_task(main.main())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if main.state.ready:
                break
        self.assertTrue(main.state.ready)                                      # health goes starting -> ready
        client = stubs.Client.instances[-1]
        self.assertTrue(client.started)
        names = [h.callback.__name__ for h in client.handlers]
        self.assertEqual(names, ["on_command", "on_user_command", "on_message"])   # commands before generic handler
        self.assertEqual((await check_health(main.state))["status"], "healthy")
        # config values reached the Mongo client
        mongo = sys.modules["motor.motor_asyncio"].AsyncIOMotorClient.instances[-1]
        self.assertEqual(mongo.kw["serverSelectionTimeoutMS"], main.DB_TIMEOUT_MS)

        event.set()                                                            # SIGTERM -> idle() returns
        await asyncio.wait_for(runner_task, 5)
        self.assertTrue(client.stopped)
        self.assertTrue(mongo.closed)
        self.assertTrue(sys.modules["aiohttp.web"].AppRunner.instances[-1].cleaned)
        self.assertEqual((await check_health(main.state))["status"], "stopping")


class TestConfig(unittest.TestCase):
    BASE = dict(API_ID="1", API_HASH="h", STRING_SESSION="s", MONGO_URI="mongodb://u:pw@h/", DEEPSEEK_API_KEY="k")

    def run_config(self, **env):
        full = {k: v for k, v in os.environ.items() if not k.startswith(("API_", "FALLBACK", "DEEPSEEK", "MAX_", "STICKER_", "PORT"))}
        full.update(self.BASE)
        full.update(env)
        for k in [k for k, v in full.items() if v is None]:
            del full[k]
        return subprocess.run([sys.executable, "-c", "import config; print('OK', len(config.PROVIDER_CONFIGS))"],
                              cwd=ROOT, env=full, capture_output=True, text=True)

    def test_valid_minimal_config(self):
        r = self.run_config()
        self.assertEqual(r.stdout.strip(), "OK 1")

    def test_missing_required_variables_are_listed(self):
        r = self.run_config(API_HASH=None, MONGO_URI=None)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Missing required environment variables: API_HASH, MONGO_URI", r.stderr)

    def test_invalid_numbers_are_reported_by_name(self):
        r = self.run_config(MAX_TOKENS="abc", STICKER_REPLY_CHANCE="2", PORT="99999", REPLY_DELAY="-1")
        self.assertNotEqual(r.returncode, 0)
        for fragment in ("MAX_TOKENS must be a whole number", "STICKER_REPLY_CHANCE must be <= 1.0",
                         "PORT must be <= 65535", "REPLY_DELAY must be >= 0.0"):
            self.assertIn(fragment, r.stderr)

    def test_incomplete_fallback_and_bad_url_and_bad_json(self):
        r = self.run_config(FALLBACK_URL="https://x/v1/chat/completions", DEEPSEEK_URL="ftp://nope", DEEPSEEK_EXTRA="{bad")
        self.assertIn("FALLBACK_URL, FALLBACK_API_KEY and FALLBACK_MODEL must all be set together", r.stderr)
        self.assertIn("URL must start with http", r.stderr)
        self.assertIn("DEEPSEEK_EXTRA must be a JSON object", r.stderr)

    def test_quotes_around_mongo_uri_are_tolerated_and_secrets_collected(self):
        r = subprocess.run(
            [sys.executable, "-c", "import config; print(config.MONGO_URI); print('pw' in config.SECRETS)"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, **self.BASE, "MONGO_URI": '"mongodb://u:pw@h/"'})
        self.assertEqual(r.stdout.split(), ["mongodb://u:pw@h/", "True"])

    def test_existing_environment_variable_names_are_unchanged(self):
        r = self.run_config(DEEPSEEK_URL="https://api.groq.com/openai/v1/chat/completions", DEEPSEEK_MODEL="openai/gpt-oss-120b",
                            FALLBACK_URL="https://a/v1", FALLBACK_API_KEY="k2", FALLBACK_MODEL="m",
                            FALLBACK2_URL="https://b/v1", FALLBACK2_API_KEY="k3", FALLBACK2_MODEL="m2",
                            HISTORY_LIMIT="10", MEMORY_DAYS="90", PERSONA_NAME="Cherry", REPLY_DELAY="3")
        self.assertEqual(r.stdout.strip(), "OK 3")


class TestDashboardConfig(unittest.TestCase):
    BASE = TestConfig.BASE

    def probe(self, **env):
        full = {k: v for k, v in os.environ.items() if not k.startswith(("API_", "FALLBACK", "DEEPSEEK", "ADMIN", "CHAT_", "LOG_"))}
        full.update(self.BASE)
        full.update({k: v for k, v in env.items() if v is not None})
        return subprocess.run(
            [sys.executable, "-c", "import config; print(config.CHAT_LOG, config.ADMIN_PASSWORD in config.SECRETS if config.ADMIN_PASSWORD else '-')"],
            cwd=ROOT, env=full, capture_output=True, text=True)

    def test_dashboard_and_chat_log_are_off_without_a_password(self):
        self.assertEqual(self.probe().stdout.split(), ["False", "-"])
        self.assertEqual(self.probe(CHAT_LOG="1").stdout.split(), ["False", "-"])        # never logs chats for a dashboard nobody can open

    def test_password_enables_both_and_is_treated_as_a_secret(self):
        self.assertEqual(self.probe(ADMIN_PASSWORD="long-enough-pass").stdout.split(), ["True", "True"])
        self.assertEqual(self.probe(ADMIN_PASSWORD="long-enough-pass", CHAT_LOG="0").stdout.split()[0], "False")

    def test_short_password_is_refused_at_startup(self):
        r = self.probe(ADMIN_PASSWORD="short")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ADMIN_PASSWORD must be at least 8 characters", r.stderr)


class TestRedaction(unittest.TestCase):
    def fmt(self):
        return RedactingFormatter("%(message)s", main.SECRETS)

    def test_configured_secrets_are_masked_everywhere(self):
        f = self.fmt()
        for secret in main.SECRETS:
            rec = logging.LogRecord("x", logging.ERROR, __file__, 1, f"boom {secret} end", None, None)
            self.assertNotIn(secret, f.format(rec))

    def test_exception_tracebacks_are_masked_too(self):
        f = self.fmt()
        try:
            raise RuntimeError(f"auth failed for {main.SECRETS[1]}")
        except RuntimeError:
            rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
        self.assertNotIn(main.SECRETS[1], f.format(rec))

    def test_key_shaped_strings_and_db_passwords_are_masked_even_if_unknown(self):
        f = RedactingFormatter("%(message)s", [])
        for text in ["key sk-abcdefghijklmnop1234", "Bearer abcdefghijklmnop1234", "uri mongodb+srv://user:hunter2pw@cluster0.x.mongodb.net/db",
                     "gsk_abcdefghijklmnopqrstuv"]:
            rec = logging.LogRecord("x", logging.INFO, __file__, 1, text, None, None)
            out = f.format(rec)
            self.assertNotIn("abcdefghijklmnop", out)
            self.assertNotIn("hunter2pw", out)

    def test_normal_messages_are_untouched(self):
        rec = logging.LogRecord("x", logging.INFO, __file__, 1, "Loaded 54 stickers from 2 packs", None, None)
        self.assertEqual(self.fmt().format(rec), "Loaded 54 stickers from 2 packs")


class TestBoundedContainers(unittest.IsolatedAsyncioTestCase):
    def test_bounded_dict_and_set(self):
        d = BoundedDict(3)
        for i in range(10):
            d[i] = i
        self.assertEqual(list(d), [7, 8, 9])
        s = BoundedSet(2)
        self.assertTrue(s.add_if_new(1))
        self.assertFalse(s.add_if_new(1))
        s.add_if_new(2); s.add_if_new(3)
        self.assertEqual(len(s), 2)
        self.assertTrue(s.add_if_new(1))                                       # forgotten, so new again

    async def test_lock_pool_stays_bounded_but_never_evicts_a_held_lock(self):
        pool = LockPool(3)
        held = pool.get("held")
        await held.acquire()
        for i in range(20):
            pool.get(i)
        self.assertLessEqual(len(pool), 4)
        self.assertIs(pool.get("held"), held)                                  # still the same lock object
        held.release()
