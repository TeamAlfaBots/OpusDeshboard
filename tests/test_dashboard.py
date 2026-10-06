"""Owner dashboard: auth, rate limit, security headers, masking, flags, filters, logging from the bot."""
import asyncio
import re
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import dashboard
import main
import memory as memmod
from dashboard import COOKIE, Dashboard
from health import HealthState, make_app
from memory import FLAGS, Memory, detect_flags, mask_pii
from tests.fakes import FakeClient, FakeDB, FakeMessage, make_sticker

PASSWORD = "test-admin-pass-123"


def req(*, cookie=None, query=None, form=None, ip="1.1.1.1", headers=None, secure=False):
    async def post(): return form or {}
    return SimpleNamespace(cookies={COOKIE: cookie} if cookie else {}, query=query or {}, post=post,
                           headers=headers or {}, remote=ip, secure=secure)


class DashCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = FakeDB()
        self.mem = Memory(self.db)
        self.dash = Dashboard(lambda: self.mem, password=PASSWORD)
        self.cookie = self.dash.make_cookie()

    async def add(self, text="hi", reply="hello", uid=1, name="Aman", chat_id=None, **kw):
        await self.mem.log_turn(chat_id=chat_id if chat_id is not None else uid, chat_type=kw.pop("chat_type", "private"),
                                chat_title=kw.pop("chat_title", None), user_id=uid, name=name, username=f"user{uid}",
                                user_text=text, bot_text=reply, **kw)


class TestAuth(DashCase):
    async def test_no_password_means_no_routes_at_all(self):
        app = make_app(HealthState(), setup=lambda a: dashboard.setup(a, lambda: self.mem, password=""))
        self.assertEqual(set(app.router.routes), {"/", "/health"})
        self.assertEqual(app.router.post_routes, {})

    async def test_routes_are_registered_next_to_health(self):
        app = make_app(HealthState(), setup=lambda a: dashboard.setup(a, lambda: self.mem, password=PASSWORD))
        self.assertEqual(set(app.router.routes), {"/", "/health", "/admin", "/admin/api/users", "/admin/api/messages"})
        self.assertEqual(set(app.router.post_routes), {"/admin/login", "/admin/logout"})

    async def test_unauthenticated_page_is_the_login_form_without_data(self):
        r = await self.dash.page(req())
        self.assertEqual(r.status, 200)
        self.assertIn('name="password"', r.text)
        self.assertNotIn("/admin/api", r.text)

    async def test_api_requires_login(self):
        for handler in (self.dash.api_users, self.dash.api_messages):
            r = await handler(req())
            self.assertEqual(r.status, 401)
        r = await self.dash.api_users(req(cookie="123.deadbeef"))
        self.assertEqual(r.status, 401)

    async def test_wrong_password_rejected_without_hint_or_cookie(self):
        r = await self.dash.login(req(form={"password": "nope"}))
        self.assertEqual(r.status, 401)
        self.assertEqual(r.cookies, {})
        self.assertIn("Wrong password", r.text)

    async def test_correct_password_sets_hardened_cookie(self):
        r = await self.dash.login(req(form={"password": PASSWORD}, headers={"X-Forwarded-Proto": "https"}))
        self.assertEqual((r.status, r.headers["Location"]), (303, "/admin"))
        c = r.cookies[COOKIE]
        self.assertTrue(c["httponly"] and c["secure"])
        self.assertEqual(c["samesite"], "Strict")
        self.assertEqual(c["path"], "/admin")
        page = await self.dash.page(req(cookie=c["value"]))
        self.assertIn("Chat monitor", page.text)
        self.assertIn("/admin/api/users", page.text)

    async def test_cookie_is_not_marked_secure_on_plain_http(self):
        r = await self.dash.login(req(form={"password": PASSWORD}))
        self.assertFalse(r.cookies[COOKIE]["secure"])

    async def test_tampered_expired_and_foreign_cookies_are_rejected(self):
        d = self.dash
        self.assertTrue(d.valid_cookie(d.make_cookie()))
        exp, sig = d.make_cookie().split(".")
        self.assertFalse(d.valid_cookie(f"{int(exp) + 999}.{sig}"))                     # changed expiry
        self.assertFalse(d.valid_cookie(f"{exp}.{'0' * len(sig)}"))
        self.assertFalse(d.valid_cookie(d.make_cookie(now=time.time() - 10 * 3600 * 24)))  # expired
        self.assertFalse(d.valid_cookie(None))
        self.assertFalse(d.valid_cookie("garbage"))
        other = Dashboard(lambda: self.mem, password="another-password-1")
        self.assertFalse(d.valid_cookie(other.make_cookie()))                            # other password = other key

    async def test_brute_force_is_rate_limited_even_for_the_right_password(self):
        for _ in range(self.dash.max_attempts):
            await self.dash.login(req(form={"password": "bad"}, ip="9.9.9.9"))
        locked = await self.dash.login(req(form={"password": PASSWORD}, ip="9.9.9.9"))
        self.assertEqual(locked.status, 429)
        self.assertEqual(locked.cookies, {})
        other_ip = await self.dash.login(req(form={"password": PASSWORD}, ip="8.8.8.8"))
        self.assertEqual(other_ip.status, 303)
        # window passes -> allowed again
        self.assertFalse(self.dash.locked("9.9.9.9", now=time.time() + self.dash.window + 1))

    async def test_forwarded_ip_is_used_behind_the_proxy(self):
        self.assertEqual(Dashboard.client_ip(req(headers={"X-Forwarded-For": "5.5.5.5, 10.0.0.1"})), "5.5.5.5")

    async def test_logout_clears_cookie(self):
        r = await self.dash.logout(req(cookie=self.cookie))
        self.assertIn(COOKIE, r.deleted_cookies)

    async def test_security_headers(self):
        for r in (await self.dash.page(req()), await self.dash.page(req(cookie=self.cookie)), await self.dash.api_users(req(cookie=self.cookie))):
            h = r.headers
            self.assertEqual(h["Cache-Control"], "no-store")
            self.assertIn("noindex", h["X-Robots-Tag"])
            self.assertEqual(h["X-Frame-Options"], "DENY")
            self.assertIn("frame-ancestors 'none'", h["Content-Security-Policy"])
            self.assertNotIn("unsafe-inline", h["Content-Security-Policy"])
        page = await self.dash.page(req(cookie=self.cookie))
        nonce = re.search(r"nonce-([\w-]+)", page.headers["Content-Security-Policy"]).group(1)
        self.assertEqual(page.text.count(f'nonce="{nonce}"'), 2)                         # inline <style> and <script> only

    async def test_page_never_uses_innerhtml_so_chat_text_cannot_run_script(self):
        page = (await self.dash.page(req(cookie=self.cookie))).text
        self.assertNotIn("innerHTML", page)
        self.assertNotIn("insertAdjacentHTML", page)
        self.assertNotIn("document.write", page)
        self.assertIn("textContent", page)


class TestMaskingAndFlags(DashCase):
    def test_mask_pii(self):
        out = mask_pii("call 9876543210 or a.b@gmail.com, otp is 123456, +91 98765 43210, sk-abcdefghijklmnop1234, Bearer abcdefghijklmnop1234")
        for raw in ("9876543210", "a.b@gmail.com", "123456", "98765 43210", "sk-abcdefghijklmnop", "Bearer abcdef"):
            self.assertNotIn(raw, out)
        self.assertIn("[number]", out)
        self.assertIn("[email]", out)

    def test_mask_keeps_normal_text(self):
        self.assertEqual(mask_pii("kal 10:30 baje milte hain, 12/05/2026"), "kal 10:30 baje milte hain, 12/05/2026")

    async def test_database_never_contains_raw_private_data(self):
        await self.add("mera number 9876543210 hai, mail a@b.com, otp 445566", "ok number [x] 9123456780")
        stored = str(self.db.chat_log.docs) + str(self.db.chat_users.docs)
        for raw in ("9876543210", "a@b.com", "445566", "9123456780"):
            self.assertNotIn(raw, stored)

    def test_flags(self):
        cases = {
            "real?": ["tum real ho?", "are you a bot", "AI ho kya tum", "fake id hai kya", "tum bot ho"],
            "meet": ["milna hai tumse", "number do na", "video call karte hain", "photo bhej", "whatsapp pe aao"],
            "love": ["i love you", "tumse pyar ho gaya", "miss you jaan", "will you marry me", "tum cute ho"],
            "money": ["paise bhej do", "upi id do", "recharge karwa do", "send money"],
            "minor?": ["main 15 saal ka hu", "i am in class 9", "school se aaya", "age is 14"],
            "distress": ["mujhe suicide ke khayal aate hain", "i want to kill myself", "jeena nahi chahta"],
        }
        for flag, texts in cases.items():
            for t in texts:
                with self.subTest(flag=flag, text=t):
                    self.assertIn(flag, detect_flags(t))

    def test_plain_chat_has_no_flags_and_bot_side_love_is_detected(self):
        for t in ["kya haal hai", "aaj movie dekhi", "khana khaya?", "hmm", "mera exam hai kal"]:
            self.assertEqual(detect_flags(t), [], t)
        self.assertEqual(detect_flags("hi", "aww cute ho tum"), ["love"])               # flirty reply is flagged
        self.assertEqual(detect_flags("tum real ho?", "haan"), ["real?"])
        self.assertEqual(set(FLAGS), {"love", "real?", "meet", "minor?", "distress", "money"})


class TestLogAndQueries(DashCase):
    async def test_users_summary_counts_and_flags(self):
        await self.add("hello", "hi", uid=1, name="Aman")
        await self.add("i love you", "aww", uid=1, name="Aman")
        await self.add("hey", "hi", uid=2, name="Riya")
        users = {u["user_id"]: u for u in await self.mem.dash_users()}
        self.assertEqual(users[1]["count"], 2)
        self.assertEqual(users[1]["flags"], ["love"])
        self.assertEqual(users[1]["last_text"], "i love you")
        self.assertEqual(users[2]["flags"], [])

    async def test_users_filter_and_search(self):
        await self.add("i love you", uid=1, name="Aman")
        await self.add("hello", uid=2, name="Priya")
        self.assertEqual([u["user_id"] for u in await self.mem.dash_users(flag="love")], [1])
        self.assertEqual([u["user_id"] for u in await self.mem.dash_users(q="priya")], [2])
        self.assertEqual(await self.mem.dash_users(q="nobody"), [])

    async def test_messages_filters_by_user_chat_flag_and_text(self):
        await self.add("hello there", "hi", uid=1)
        await self.add("tum real ho?", "ai hu main", uid=1)
        await self.add("cricket dekha", "haan", uid=2)
        await self.add("hey group", "hi", uid=1, chat_id=-100, chat_type="group", chat_title="Friends")
        msgs, _ = await self.mem.dash_messages(user_id=1, chat_id=1)
        self.assertEqual({m["user_text"] for m in msgs}, {"hello there", "tum real ho?"})
        msgs, _ = await self.mem.dash_messages(flag="real?")
        self.assertEqual([m["user_text"] for m in msgs], ["tum real ho?"])
        msgs, _ = await self.mem.dash_messages(q="CRICKET")
        self.assertEqual([m["user_text"] for m in msgs], ["cricket dekha"])
        msgs, _ = await self.mem.dash_messages(q="ai hu")                               # also searches the bot's replies
        self.assertEqual(len(msgs), 1)
        msgs, _ = await self.mem.dash_messages(chat_id=-100)
        self.assertEqual((msgs[0]["chat_title"], msgs[0]["chat_type"]), ("Friends", "group"))

    async def test_search_text_is_treated_literally_not_as_a_regex(self):
        await self.add("price is 5+5 (ok)", uid=1)
        await self.add("something else", uid=2)
        msgs, _ = await self.mem.dash_messages(q=".*")
        self.assertEqual(msgs, [])
        msgs, _ = await self.mem.dash_messages(q="5+5 (ok)")
        self.assertEqual(len(msgs), 1)

    async def test_pagination_newest_first_with_before_cursor(self):
        for i in range(5):
            await self.add(f"m{i}", uid=1)
            await asyncio.sleep(0.002)
        page1, nb = await self.mem.dash_messages(limit=2)
        self.assertEqual([m["user_text"] for m in page1], ["m4", "m3"])
        self.assertIsNotNone(nb)
        page2, nb2 = await self.mem.dash_messages(limit=2, before=datetime.fromisoformat(nb))
        self.assertEqual([m["user_text"] for m in page2], ["m2", "m1"])
        page3, nb3 = await self.mem.dash_messages(limit=2, before=datetime.fromisoformat(nb2))
        self.assertEqual([m["user_text"] for m in page3], ["m0"])
        self.assertIsNone(nb3)

    async def test_log_turn_never_raises_when_database_is_down(self):
        self.db.fail = True
        await self.add("hello", "hi")                                                    # must not raise
        self.assertIsNone(await self.mem.dash_users())
        self.assertIsNone(await self.mem.dash_messages())

    async def test_ttl_indexes_for_log_but_still_none_for_long_memory(self):
        await self.mem.ensure_indexes()
        for coll in ("chat_log", "chat_users"):
            ttl = [kw for _s, kw in getattr(self.db, coll).indexes if "expireAfterSeconds" in kw]
            self.assertEqual(ttl[0]["expireAfterSeconds"], memmod.LOG_DAYS * 86400, coll)
        self.assertFalse(any("expireAfterSeconds" in kw for _s, kw in self.db.long_memory.indexes))

    async def test_clear_log_removes_only_that_user_in_that_chat(self):
        await self.add("a", uid=1)
        await self.add("b", uid=2)
        await self.mem.clear_log(1, 1)
        self.assertEqual([u["user_id"] for u in await self.mem.dash_users()], [2])
        msgs, _ = await self.mem.dash_messages()
        self.assertEqual([m["user_id"] for m in msgs], [2])


class TestApi(DashCase):
    async def test_users_and_messages_endpoints(self):
        await self.add("tum real ho?", "ai hu main", uid=7, name="Karan")
        r = await self.dash.api_users(req(cookie=self.cookie))
        self.assertEqual((r.status, r.data["users"][0]["name"], r.data["users"][0]["flags"]), (200, "Karan", ["real?"]))
        r = await self.dash.api_messages(req(cookie=self.cookie, query={"chat_id": "7", "user_id": "7"}))
        self.assertEqual(r.status, 200)
        self.assertEqual(r.data["messages"][0]["bot_text"], "ai hu main")
        self.assertIsNone(r.data["next_before"])

    async def test_bad_parameters_are_rejected_not_executed(self):
        c = self.cookie
        for handler, query in [(self.dash.api_messages, {"chat_id": "abc"}), (self.dash.api_messages, {"limit": "x"}),
                               (self.dash.api_messages, {"before": "yesterday"}), (self.dash.api_messages, {"flag": "$ne"}),
                               (self.dash.api_users, {"flag": "<script>"})]:
            r = await handler(req(cookie=c, query=query))
            self.assertEqual(r.status, 400, query)

    async def test_limit_is_clamped(self):
        for i in range(5):
            await self.add(f"m{i}", uid=1)
        r = await self.dash.api_messages(req(cookie=self.cookie, query={"limit": "999999"}))
        self.assertEqual(r.status, 200)
        r = await self.dash.api_messages(req(cookie=self.cookie, query={"limit": "-5"}))
        self.assertEqual(len(r.data["messages"]), 1)                                     # clamped to >= 1

    async def test_script_in_a_chat_message_is_returned_as_inert_data(self):
        await self.add("<script>alert(1)</script><img src=x onerror=alert(2)>", "ok")
        r = await self.dash.api_messages(req(cookie=self.cookie))
        self.assertEqual(r.headers["Content-Security-Policy"].count("script-src"), 0)     # JSON response: no script allowed at all
        self.assertIn("<script>", r.data["messages"][0]["user_text"])                    # data is kept verbatim, rendered via textContent

    async def test_database_down_is_reported_as_503(self):
        self.db.fail = True
        self.assertEqual((await self.dash.api_users(req(cookie=self.cookie))).status, 503)
        self.assertEqual((await self.dash.api_messages(req(cookie=self.cookie))).status, 503)

    async def test_before_startup_memory_is_none(self):
        d = Dashboard(lambda: None, password=PASSWORD)
        self.assertEqual((await d.api_users(req(cookie=d.make_cookie()))).status, 503)


class TestBotWritesTheLog(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = FakeDB()
        main.memory = Memory(self.db)
        self.client = FakeClient()
        main._seen = main.BoundedSet(100)
        main._pending.clear()
        main._fallback_sent.clear()
        main._enabled_cache.clear()
        main._locks = main.LockPool(100)
        self.replies = []

        async def noop(*a, **k): return None

        async def ask(messages, max_tokens=None, temperature=None):
            return self.replies.pop(0) if self.replies else "reply"
        for p in (mock.patch.object(main, "_sleep", noop), mock.patch.object(main, "ask_deepseek", ask)):
            p.start()
            self.addCleanup(p.stop)

    async def test_normal_turn_is_logged_with_masked_text_and_flags(self):
        self.replies = ["aww cute 😊 || kaisi ho"]
        await main.on_message(self.client, FakeMessage("tum real ho? mera no 9876543210"))
        doc = self.db.chat_log.docs[0]
        self.assertEqual(doc["user_text"], "tum real ho? mera no [number]")
        self.assertEqual(doc["bot_text"], "aww cute 😊\nkaisi ho")
        self.assertEqual((doc["status"], doc["chat_type"]), ("ok", "private"))
        self.assertEqual(set(doc["flags"]), {"real?", "love"})

    async def test_ai_down_and_send_failure_are_logged_too(self):
        self.replies = [None]
        await main.on_message(self.client, FakeMessage("hello"))
        self.assertEqual(self.db.chat_log.docs[-1]["status"], "ai_down")
        self.assertTrue(self.db.chat_log.docs[-1]["bot_text"])                           # the friendly fallback that was sent
        self.replies = ["hii"]
        bad = FakeMessage("hello again")
        bad.fail_on_send = lambda n: RuntimeError("blocked")
        await main.on_message(self.client, bad)
        self.assertEqual((self.db.chat_log.docs[-1]["status"], self.db.chat_log.docs[-1]["bot_text"]), ("send_failed", None))

    async def test_partial_send_is_marked(self):
        self.replies = ["one || two"]
        msg = FakeMessage("hello")
        msg.fail_on_send = lambda n: RuntimeError("blocked") if n == 2 else None
        await main.on_message(self.client, msg)
        self.assertEqual((self.db.chat_log.docs[0]["status"], self.db.chat_log.docs[0]["bot_text"]), ("partial", "one"))

    async def test_group_turn_has_group_type_and_sticker_turn_has_kind(self):
        await main.on_message(self.client, FakeMessage("@cherry hi", group=True, mentioned=True))
        self.assertEqual(self.db.chat_log.docs[0]["chat_type"], "group")
        main._sticker_cache.update(ts=time.monotonic(), items=[{"id": 2, "file_id": "enc_2", "emoji": "😂"}], packs=1)
        with mock.patch.object(main, "STICKER_REPLY_CHANCE", 1.0):
            await main.on_message(self.client, FakeMessage(None, sticker=make_sticker(1, "😂")))
        last = self.db.chat_log.docs[-1]
        self.assertEqual((last["kind"], last["bot_text"]), ("sticker", "[sticker 😂]"))

    async def test_messages_the_bot_does_not_handle_are_not_logged(self):
        await main.on_message(self.client, FakeMessage("untagged chatter", group=True))
        await main.on_message(self.client, FakeMessage("/start"))
        self.assertEqual(self.db.chat_log.docs, [])

    async def test_logging_is_off_when_dashboard_disabled(self):
        with mock.patch.object(memmod, "CHAT_LOG", False):
            await main.on_message(self.client, FakeMessage("hello"))
        self.assertEqual(self.db.chat_log.docs, [])

    async def test_forgetall_also_clears_the_dashboard_log(self):
        await main.on_message(self.client, FakeMessage("hello", uid=5))
        self.assertEqual(len(self.db.chat_log.docs), 1)
        cmd = FakeMessage(".forgetall", uid=5)
        await main.on_user_command(self.client, cmd)
        self.assertEqual(self.db.chat_log.docs, [])
        self.assertEqual(self.db.chat_users.docs, [])

    async def test_a_broken_database_does_not_stop_the_bot_from_replying(self):
        self.db.fail = True
        msg = FakeMessage("hello")
        await main.on_message(self.client, msg)
        self.assertEqual(msg.texts, ["reply"])
