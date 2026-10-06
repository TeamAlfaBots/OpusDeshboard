"""Telegram handlers end to end (fake Telegram, fake MongoDB, fake AI)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import main
import memory as memmod
import providers
from memory import Memory
from tests.fakes import (FakeClient, FakeDB, FakeHTTP, FakeMessage, drain, err, make_doc, make_sticker, ok)
from tests.fakes import user as mkuser
from pyrogram.errors import FloodWait

DS = "https://api.deepseek.com/chat/completions"
GQ = "https://api.groq.com/openai/v1/chat/completions"


class BotCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = FakeDB()
        main.memory = Memory(self.db)
        await main.memory.ensure_indexes()
        self.client = FakeClient()
        for cache in (main._enabled_cache, main._fallback_sent):
            cache.clear()
        main._pending.clear()
        main._seen = main.BoundedSet(2000)
        main._locks = main.LockPool(2000)
        main._sticker_cache.update(ts=0.0, items=[], packs=0, busy=False)

        async def noop(*a, **k): return None
        patches = [mock.patch.object(main, "_sleep", noop), mock.patch.object(providers, "_sleep", noop)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        # fake AI: records every call and replies from a queue (or a default)
        self.ai_calls, self.ai_replies = [], []

        async def fake_ask(messages, max_tokens=None, temperature=None):
            self.ai_calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if self.ai_replies:
                item = self.ai_replies.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return f"reply {len(self.ai_calls)}"
        p = mock.patch.object(main, "ask_deepseek", fake_ask)
        p.start()
        self.addCleanup(p.stop)

    # helpers
    async def say(self, text, **kw):
        msg = FakeMessage(text, **kw)
        await main.on_message(self.client, msg)
        return msg

    def last_system(self):
        return self.ai_calls[-1]["messages"][0]["content"]

    async def settle(self):
        await drain(list(main._bg_tasks))


class TestDirectMessages(BotCase):
    async def test_normal_dm_reply_memory_and_clean_typing(self):
        self.ai_replies = ["hii 😊 bolo"]
        msg = await self.say("hi")
        self.assertEqual(msg.sent, [("text", "hii 😊 bolo", False)])          # DM: plain message, no quote
        hist = await main.memory.load_history(42, 42)
        self.assertEqual([(m["role"], m["content"]) for m in hist], [("user", "hi"), ("assistant", "hii 😊 bolo")])
        self.assertTrue(any(a[1] == "typing" for a in self.client.actions))     # typing animation shown
        stray = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(stray, [])                                            # typing task cleaned up

    async def test_reply_is_split_into_bubbles_and_memory_matches_what_was_sent(self):
        self.ai_replies = ["arre sach me?? || kab hua ye"]
        msg = await self.say("mera exam cancel ho gaya")
        self.assertEqual(msg.texts, ["arre sach me??", "kab hua ye"])
        hist = await main.memory.load_history(42, 42)
        self.assertEqual(hist[-1]["content"], "arre sach me?? kab hua ye")

    async def test_bots_commands_and_empty_messages_are_ignored(self):
        for text, kw in [("/start", {}), (".chatoff", {}), ("   ", {}), (None, {})]:
            self.assertEqual((await self.say(text, **kw)).sent, [])
        bot_msg = FakeMessage("hello")
        bot_msg.from_user.is_bot = True
        await main.on_message(self.client, bot_msg)
        self.assertEqual(bot_msg.sent, [])
        self.assertEqual(self.ai_calls, [])
        self.assertEqual((await self.say("...")).texts, ["reply 1"])          # '...' is chat, not a command

    async def test_duplicate_delivery_of_same_message_is_processed_once(self):
        msg = FakeMessage("hello")
        await main.on_message(self.client, msg)
        await main.on_message(self.client, msg)
        self.assertEqual(len(self.ai_calls), 1)
        self.assertEqual(len(msg.sent), 1)

    async def test_flood_of_messages_is_bounded(self):
        gate = asyncio.Event()

        async def slow_ask(messages, max_tokens=None, temperature=None):
            await gate.wait()
            return "ok"
        with mock.patch.object(main, "ask_deepseek", slow_ask), mock.patch.object(main, "MAX_PENDING_PER_USER", 2):
            msgs = [FakeMessage(f"m{i}") for i in range(5)]
            tasks = [asyncio.create_task(main.on_message(self.client, m)) for m in msgs]
            await asyncio.sleep(0.05)
            gate.set()
            await asyncio.gather(*tasks)
        self.assertEqual(sum(1 for m in msgs if m.sent), 2)                    # extra messages dropped, none crashed


class TestLanguageAndContext(BotCase):
    async def feed(self, texts, **kw):
        for t in texts:
            await self.say(t, **kw)

    async def test_short_hinglish_style_is_learned(self):
        await self.feed(["kya kr rhi yrr", "bro mood off hai", "yrr tu bhi na", "bro kuch bata", "acha yrr thik", "hn bro", "kya hua yrr"])
        sysmsg = self.last_system()
        self.assertIn("Hinglish", sysmsg)
        self.assertIn("bahut chhote msgs", sysmsg)
        self.assertIn("lowercase", sysmsg)

    async def test_hindi_devanagari_is_detected(self):
        await self.feed(["आज मौसम बहुत अच्छा है", "मुझे चाय पसंद है", "तुम क्या कर रही हो", "कल मिलते हैं", "ठीक है ना", "चलो बात करते हैं"])
        self.assertIn("Devanagari", self.last_system())

    async def test_english_is_detected(self):
        await self.feed(["Hey, how was your day today?", "I was thinking about going out tonight", "Do you like watching movies on weekends?",
                         "That sounds really great honestly", "Let me know what you think about it", "Okay see you tomorrow then"])
        style = self.last_system().split("SAMNE WALE KA CHAT STYLE")[1]       # the persona prompt itself mentions Hinglish
        self.assertIn("Language: English", style)
        self.assertNotIn("Hinglish", style)

    async def test_no_style_hint_before_enough_messages(self):
        await self.say("hi")
        self.assertNotIn("CHAT STYLE", self.last_system())

    async def test_recent_context_is_sent_in_order(self):
        self.ai_replies = ["a1", "a2"]
        await self.say("pehla msg")
        await self.say("doosra msg")
        await self.say("teesra msg")
        sent = [(m["role"], m["content"]) for m in self.ai_calls[-1]["messages"][1:]]
        self.assertEqual(sent, [("user", "pehla msg"), ("assistant", "a1"), ("user", "doosra msg"),
                                ("assistant", "a2"), ("user", "teesra msg")])

    async def test_other_users_conversation_never_leaks_into_context(self):
        await self.say("secret topic alpha", uid=1)
        await self.say("hello", uid=2)
        flat = " ".join(m["content"] for m in self.ai_calls[-1]["messages"])
        self.assertNotIn("alpha", flat)

    async def test_repeated_reply_triggers_one_regeneration(self):
        self.ai_replies = ["haha pagal", "haha pagal", "alag baat bolti hu"]
        await self.say("kuch bol")
        msg = await self.say("aur bol")                       # model repeats itself -> retry with a hint
        self.assertEqual(msg.texts, ["alag baat bolti hu"])
        self.assertIn("pichle reply jaisa", self.ai_calls[-1]["messages"][0]["content"])

    async def test_previous_replies_are_shown_so_phrases_are_not_repeated(self):
        await self.say("hello")
        await self.say("hello again")
        self.assertIn("PICHLE REPLIES", self.last_system())


class TestLongTermMemoryInChat(BotCase):
    async def test_remember_then_use_only_when_relevant(self):
        await _user_cmd(self, ".remember mujhe cricket pasand hai", uid=7)
        self.assertEqual(self.ai_calls, [])                                   # command, not a chat message
        await self.say("kya haal hai", uid=7)
        self.assertNotIn("mujhe cricket pasand hai", self.last_system())      # unrelated -> not injected
        await self.say("aaj cricket match dekha", uid=7)
        self.assertIn("mujhe cricket pasand hai", self.last_system())         # relevant -> injected
        await self.say("kya kar rahi ho", uid=7)                              # topic just changed: still in the last-2 window
        await self.say("khana khaya", uid=7)
        await self.say("hmm", uid=7)
        self.assertNotIn("mujhe cricket pasand hai", self.last_system())      # conversation moved on -> dropped again

    async def test_user_command_replies(self):
        for cmd in (".remember", ".forget", ".forget abc"):
            self.assertTrue((await _user_cmd(self, cmd)).sent)               # usage hints, no crash
        m = await _user_cmd(self, ".remember mera password hunter2 hai")
        self.assertIn("save nahi", m.texts[0])
        m = await _user_cmd(self, ".remember mujhe chai pasand hai")
        self.assertIn("yaad rakh", m.texts[0])
        m = await _user_cmd(self, ".remember mujhe chai pasand hai")
        self.assertIn("pehle se", m.texts[0])

    async def test_mymemory_forget_and_forgetall(self):
        await _user_cmd(self, ".remember mujhe chai pasand hai")
        await _user_cmd(self, ".remember main guitar seekh raha hu")
        listing = (await _user_cmd(self, ".mymemory")).texts[0]
        self.assertIn("1. mujhe chai pasand hai", listing)
        self.assertIn("2. main guitar seekh raha hu", listing)
        self.assertIn("hata diya", (await _user_cmd(self, ".forget 1")).texts[0])
        self.assertEqual([f["fact"] for f in await main.memory.list_facts(42, 42)], ["main guitar seekh raha hu"])
        await self.say("hello")                                              # creates recent history
        self.assertIn("sab bhool gayi", (await _user_cmd(self, ".forgetall")).texts[0])
        self.assertEqual(await main.memory.list_facts(42, 42), [])
        self.assertEqual(await main.memory.load_history(42, 42), [])

    async def test_cross_user_isolation_in_a_group(self):
        await _user_cmd(self, ".remember mujhe cricket pasand hai", uid=1, group=True, mentioned=True)
        await self.say("cricket dekha aaj", uid=2, group=True, mentioned=True)
        self.assertNotIn("mujhe cricket pasand hai", self.last_system())
        await self.say("cricket dekha aaj", uid=1, group=True, mentioned=True)
        self.assertIn("mujhe cricket pasand hai", self.last_system())
        await self.say("cricket dekha aaj", uid=1)                           # same user, private chat = other chat
        self.assertNotIn("mujhe cricket pasand hai", self.last_system())

    async def test_automatic_extraction_stores_and_updates_facts(self):
        replies = []

        async def ask(messages, max_tokens=None, temperature=None):
            if messages[0]["content"].startswith("Tum ek memory extractor"):
                return replies.pop(0)
            return "ok"
        replies += ['{"facts":[{"key":"fav_game","fact":"user ko chess pasand hai"}]}',
                    '{"facts":[{"key":"fav_game","fact":"user ko ab carrom pasand hai"}]}']
        with mock.patch.object(main, "ask_deepseek", ask), mock.patch.object(memmod, "MEMORY_EXTRACT_EVERY", 2):
            await self.say("hello", uid=3)
            await self.say("mujhe chess pasand hai", uid=3)
            await self.settle()
            self.assertEqual([f["fact"] for f in await main.memory.list_facts(3, 3)], ["user ko chess pasand hai"])
            await self.say("ab to carrom pasand hai", uid=3)
            await self.say("haan carrom hi", uid=3)
            await self.settle()
        self.assertEqual([f["fact"] for f in await main.memory.list_facts(3, 3)], ["user ko ab carrom pasand hai"])

    async def test_explicit_remember_sentence_triggers_extraction_immediately(self):
        calls = []

        async def ask(messages, max_tokens=None, temperature=None):
            calls.append(messages[0]["content"][:20])
            return '{"facts":[{"key":"nick","fact":"user ko Appu bulana pasand hai"}]}' if "extractor" in messages[0]["content"] else "ok"
        with mock.patch.object(main, "ask_deepseek", ask):
            await self.say("yaad rakhna mujhe Appu bolna", uid=4)
            await self.settle()
        self.assertEqual([f["fact"] for f in await main.memory.list_facts(4, 4)], ["user ko Appu bulana pasand hai"])


async def _user_cmd(case, text, **kw):
    msg = FakeMessage(text, **kw)
    await main.on_user_command(case.client, msg)
    return msg


class TestDatabaseOutageInChat(BotCase):
    async def test_bot_keeps_chatting_without_mongodb(self):
        self.db.fail = True
        self.ai_replies = ["abhi bhi yahin hu"]
        msg = await self.say("hello")
        self.assertEqual(msg.texts, ["abhi bhi yahin hu"])
        self.assertFalse(main.memory.ok)
        listing = await _user_cmd(self, ".mymemory")
        self.assertIn("memory access nahi", listing.texts[0])

    async def test_memory_resumes_after_recovery(self):
        self.db.fail = True
        await self.say("pehla")
        self.db.fail = False
        await self.say("doosra")
        hist = await main.memory.load_history(42, 42)
        self.assertEqual([m["content"] for m in hist if m["role"] == "user"], ["doosra"])
        self.assertTrue(main.memory.ok)


class TestAiFailures(BotCase):
    async def test_friendly_fallback_once_per_cooldown_and_nothing_saved(self):
        self.ai_replies = [None, None, None]
        first = await self.say("hello")
        self.assertEqual(len(first.texts), 1)
        self.assertIn(first.texts[0], main.FALLBACK_TEXTS)
        second = await self.say("hello?")
        self.assertEqual(second.sent, [])                                      # no spam while AI is down
        self.assertEqual(await main.memory.load_history(42, 42), [])           # failed turns are not memorised
        with mock.patch.object(main, "FALLBACK_REPLY_COOLDOWN", 0):
            third = await self.say("hello!!")
        self.assertEqual(len(third.texts), 1)

    async def test_fallback_can_be_disabled(self):
        self.ai_replies = [None]
        with mock.patch.object(main, "FRIENDLY_FALLBACK", False):
            self.assertEqual((await self.say("hello")).sent, [])

    async def test_unexpected_exception_in_ai_layer_does_not_crash_the_handler(self):
        self.ai_replies = [RuntimeError("boom")]
        msg = await self.say("hello")
        self.assertEqual(msg.sent, [])
        self.assertEqual((await self.say("hello again")).texts, ["reply 2"])    # next message works normally


class TestProviderFailoverThroughTheBot(unittest.IsolatedAsyncioTestCase):
    """Real provider chain + fake HTTP: DeepSeek fails, Groq answers, the user gets the reply."""
    async def test_primary_429_then_backup_reply_reaches_user_and_memory(self):
        http = FakeHTTP()
        providers.set_http(http)
        main.memory = Memory(FakeDB())
        main._seen = main.BoundedSet(100)
        for p in providers.PROVIDERS:
            p.cooldown_until, p.strip_extras = 0.0, False
        http.queue(DS, err(429, "rate limited"))
        http.queue(GQ, ok("backup reply 😊"))
        client = FakeClient()
        msg = FakeMessage("hello")
        with mock.patch.object(main, "ask_deepseek", providers.ask_deepseek):
            await main.on_message(client, msg)
        self.assertEqual(msg.texts, ["backup reply 😊"])
        self.assertEqual([m["content"] for m in await main.memory.load_history(42, 42)], ["hello", "backup reply 😊"])
        for p in providers.PROVIDERS:
            p.cooldown_until = 0.0


class TestTelegramFailures(BotCase):
    async def test_short_floodwait_is_waited_out_and_message_sent(self):
        self.ai_replies = ["hii"]
        msg = FakeMessage("hello")
        msg.fail_on_send = lambda n: FloodWait(value=2) if n == 1 else None
        await main.on_message(self.client, msg)
        self.assertEqual(msg.texts, ["hii"])
        self.assertEqual(len(await main.memory.load_history(42, 42)), 2)

    async def test_long_floodwait_does_not_crash_and_nothing_is_saved(self):
        self.ai_replies = ["hii"]
        msg = FakeMessage("hello")
        msg.fail_on_send = lambda n: FloodWait(value=3600)
        await main.on_message(self.client, msg)
        self.assertEqual(msg.sent, [])
        self.assertEqual(await main.memory.load_history(42, 42), [])           # user never saw it -> not remembered

    async def test_partial_send_saves_only_what_was_delivered(self):
        self.ai_replies = ["pehli baat || doosri baat"]
        msg = FakeMessage("hello")
        msg.fail_on_send = lambda n: RuntimeError("chat write forbidden") if n == 2 else None
        await main.on_message(self.client, msg)
        self.assertEqual(msg.texts, ["pehli baat"])
        hist = await main.memory.load_history(42, 42)
        self.assertEqual(hist[-1]["content"], "pehli baat")

    async def test_typing_action_failures_are_harmless(self):
        async def bad(*a): raise RuntimeError("no rights")
        self.client.send_chat_action = bad
        self.ai_replies = ["ok ji"]
        self.assertEqual((await self.say("hello")).texts, ["ok ji"])


class TestGroupMentions(BotCase):
    async def test_untagged_group_messages_get_no_reply(self):
        msg = await self.say("hello everyone", group=True)
        self.assertEqual(msg.sent, [])
        self.assertEqual(self.ai_calls, [])

    async def test_mention_flag_replies_with_quote_and_strips_the_tag(self):
        msg = await self.say("@cherry kya haal hai", group=True, mentioned=True)
        self.assertEqual(msg.sent, [("text", "reply 1", True)])
        self.assertEqual(self.ai_calls[0]["messages"][-1]["content"], "kya haal hai")
        self.assertIn("group", self.last_system())

    async def test_username_in_text_without_flag(self):
        self.assertTrue((await self.say("hey @Cherry sun", group=True)).sent)

    async def test_reply_to_bots_message(self):
        mine = SimpleNamespace(from_user=SimpleNamespace(id=1))
        self.assertTrue((await self.say("haan wahi", group=True, reply_to=mine)).sent)

    async def test_reply_to_someone_else_is_ignored(self):
        other = SimpleNamespace(from_user=SimpleNamespace(id=99))
        self.assertEqual((await self.say("tu sun", group=True, reply_to=other)).sent, [])

    async def test_tagging_another_person_is_ignored(self):
        self.assertEqual((await self.say("@someoneelse kaise ho", group=True)).sent, [])

    async def test_text_mention_entity_of_our_account(self):
        ent = SimpleNamespace(user=SimpleNamespace(id=1))
        self.assertTrue((await self.say("hey you", group=True, entities=[ent])).sent)

    async def test_group_commands_for_users_need_a_mention(self):
        self.assertEqual((await _user_cmd(self, ".mymemory", group=True)).sent, [])
        self.assertTrue((await _user_cmd(self, ".mymemory", group=True, mentioned=True)).sent)

    async def test_tag_with_only_the_username_still_answers(self):
        msg = await self.say("@cherry", group=True, mentioned=True)
        self.assertEqual(self.ai_calls[0]["messages"][-1]["content"], "hey")


class TestChatOnOff(BotCase):
    async def owner(self, cmd, chat_id, group=True):
        msg = FakeMessage(f".{cmd}", chat_id=chat_id, group=group, command=[cmd])
        await main.on_command(self.client, msg)
        return msg

    async def test_off_is_per_chat_and_on_restores_it(self):
        off = await self.owner("chatoff", -1001)
        self.assertIn("OFF", off.edits[0])
        self.assertTrue(off.deleted)
        self.assertEqual((await self.say("hi", group=True, chat_id=-1001, mentioned=True)).sent, [])   # disabled group
        self.assertTrue((await self.say("hi", group=True, chat_id=-1002, mentioned=True)).sent)        # other group still on
        self.assertTrue((await self.say("hi", uid=5)).sent)                                            # DMs still on
        on = await self.owner("chaton", -1001)
        self.assertIn("ON", on.edits[0])
        self.assertTrue((await self.say("hi", group=True, chat_id=-1001, mentioned=True)).sent)

    async def test_setting_is_persisted_and_survives_cache_loss(self):
        await self.owner("chatoff", 555, group=False)
        main._enabled_cache.clear()
        self.assertEqual((await self.say("hi", uid=555)).sent, [])

    async def test_db_failure_still_applies_for_this_run_and_tells_owner(self):
        self.db.fail = True
        msg = await self.owner("chatoff", -1001)
        self.assertIn("DB save fail", msg.edits[0])
        self.assertEqual((await self.say("hi", group=True, chat_id=-1001, mentioned=True)).sent, [])

    async def test_db_down_defaults_to_on_for_unknown_chats(self):
        self.db.fail = True
        self.assertTrue((await self.say("hi", uid=77)).sent)


class TestStickers(BotCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.client.sticker_sets = {10: [make_doc(1, "😂"), make_doc(2, "😂"), make_doc(3, "❤️")], 11: [make_doc(4, "😭")]}
        await main.load_my_stickers(self.client, force=True)
        p = mock.patch.object(main, "STICKER_REPLY_CHANCE", 1.0)
        p.start()
        self.addCleanup(p.stop)

    async def test_packs_are_loaded(self):
        self.assertEqual(len(main._sticker_cache["items"]), 4)
        self.assertEqual(main._sticker_cache["packs"], 2)

    async def sticker(self, media_id, emoji, **kw):
        msg = FakeMessage(None, sticker=make_sticker(media_id, emoji), **kw)
        await main.on_message(self.client, msg)
        return msg

    async def test_matching_emoji_gets_a_different_sticker(self):
        for _ in range(15):
            msg = await self.sticker(1, "😂")
            self.assertEqual(msg.sent, [("sticker", "enc_2", False)])
            main._seen = main.BoundedSet(100)
        self.assertEqual(self.ai_calls, [])                                    # no AI call needed

    async def test_variation_selector_does_not_break_matching(self):
        self.assertEqual((await self.sticker(900, "❤")).sent[0][1], "enc_3")

    async def test_no_match_falls_back_to_text_with_sticker_context(self):
        msg = await self.sticker(901, "🔥")
        self.assertEqual(msg.sent[0][0], "text")
        self.assertEqual(self.ai_calls[0]["messages"][-1]["content"], "[sticker: 🔥]")
        self.assertIn("sticker bheja", self.last_system())
        with mock.patch.object(main, "STICKER_ANY", True):
            main._seen = main.BoundedSet(100)
            self.assertEqual((await self.sticker(901, "🔥")).sent[0][0], "sticker")

    async def test_expired_file_reference_falls_back_to_text_and_refreshes(self):
        msg = FakeMessage(None, sticker=make_sticker(1, "😂"))
        msg.fail_on_send = lambda n: RuntimeError("FILE_REFERENCE_EXPIRED") if n == 1 else None
        before = self.client.invoked.count("GetAllStickers")
        await main.on_message(self.client, msg)
        await self.settle()
        self.assertEqual(msg.sent[0][0], "text")
        self.assertEqual(self.client.invoked.count("GetAllStickers"), before + 1)

    async def test_chance_zero_always_text(self):
        with mock.patch.object(main, "STICKER_REPLY_CHANCE", 0.0):
            self.assertEqual((await self.sticker(1, "😂")).sent[0][0], "text")

    async def test_sticker_flood_wait_is_waited_out(self):
        msg = FakeMessage(None, sticker=make_sticker(1, "😂"))
        msg.fail_on_send = lambda n: FloodWait(value=1) if n == 1 else None
        await main.on_message(self.client, msg)
        self.assertEqual(msg.sent, [("sticker", "enc_2", False)])

    async def test_group_sticker_needs_a_mention_or_reply(self):
        self.assertEqual((await self.sticker(1, "😂", group=True)).sent, [])
        mine = SimpleNamespace(from_user=SimpleNamespace(id=1))
        self.assertTrue((await self.sticker(1, "😂", group=True, reply_to=mine)).sent)

    async def test_owner_stickers_command_reloads_packs(self):
        self.client.sticker_sets[12] = [make_doc(5, "🔥")]
        msg = FakeMessage(".stickers", command=["stickers"])
        await main.on_command(self.client, msg)
        self.assertIn("5 stickers loaded (3 packs)", msg.edits[0])

    async def test_failed_pack_loading_keeps_old_list(self):
        async def boom(req): raise RuntimeError("rpc error")
        self.client.invoke = boom
        items = await main.load_my_stickers(self.client, force=True)
        self.assertEqual(len(items), 4)
        self.assertFalse(main._sticker_cache["busy"])
