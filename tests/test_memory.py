"""Two-layer memory: recent history, long-term facts, safety filter, extraction, isolation, DB outages."""
import unittest
from unittest import mock

import memory as memmod
from memory import Memory, is_safe_fact, parse_extraction, format_facts_block
from tests.fakes import FakeDB


class MemCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = FakeDB()
        self.mem = Memory(self.db)


class TestRecentHistory(MemCase):
    async def test_order_and_roles(self):
        for i in range(3):
            await self.mem.save_turn(1, 1, f"u{i}", f"a{i}")
        hist = await self.mem.load_history(1, 1)
        self.assertEqual([m["content"] for m in hist], ["u0", "a0", "u1", "a1", "u2", "a2"])
        self.assertEqual([m["role"] for m in hist][:2], ["user", "assistant"])

    async def test_only_this_chat_and_user(self):
        await self.mem.save_turn(1, 1, "mine", "reply")
        await self.mem.save_turn(1, 2, "other user", "x")
        await self.mem.save_turn(2, 1, "other chat", "x")
        self.assertEqual([m["content"] for m in await self.mem.load_history(1, 1)], ["mine", "reply"])

    async def test_limit_is_respected(self):
        with mock.patch.object(memmod, "HISTORY_LIMIT", 4):
            for i in range(5):
                await self.mem.save_turn(1, 1, f"u{i}", f"a{i}")
            hist = await self.mem.load_history(1, 1)
        self.assertEqual([m["content"] for m in hist], ["u3", "a3", "u4", "a4"])

    async def test_malformed_and_empty_rows_are_filtered(self):
        await self.mem.save_turn(1, 1, "ok", "fine")
        col, base = self.db.memory, memmod.now()
        col.docs += [
            {"chat_id": 1, "user_id": 1, "role": "system", "content": "bad role", "ts": base},
            {"chat_id": 1, "user_id": 1, "role": "user", "content": "   ", "ts": base},
            {"chat_id": 1, "user_id": 1, "role": "user", "content": None, "ts": base},
            {"chat_id": 1, "user_id": 1, "role": "assistant", "ts": base},
        ]
        hist = await self.mem.load_history(1, 1)
        self.assertEqual([m["content"] for m in hist], ["ok", "fine"])

    async def test_context_size_is_bounded_oldest_dropped(self):
        with mock.patch.object(memmod, "MAX_CONTEXT_CHARS", 300):
            for i in range(10):
                await self.mem.save_turn(1, 1, f"user message number {i} " + "x" * 30, f"assistant {i} " + "y" * 30)
            hist = await self.mem.load_history(1, 1)
        self.assertLessEqual(sum(len(m["content"]) for m in hist), 300)
        self.assertEqual(hist[0]["role"], "user")
        self.assertIn("9", hist[-1]["content"])

    async def test_empty_turn_is_not_saved(self):
        self.assertFalse(await self.mem.save_turn(1, 1, "hi", "  "))
        self.assertEqual(self.db.memory.docs, [])

    async def test_ttl_only_on_recent_memory_never_on_long_memory(self):
        await self.mem.ensure_indexes()
        ttl = [kw for _s, kw in self.db.memory.indexes if "expireAfterSeconds" in kw]
        self.assertEqual(len(ttl), 1)
        self.assertEqual(ttl[0]["expireAfterSeconds"], memmod.MEMORY_DAYS * 86400)
        self.assertFalse(any("expireAfterSeconds" in kw for _s, kw in self.db.long_memory.indexes))
        self.assertTrue(any(kw.get("unique") for _s, kw in self.db.long_memory.indexes))

    async def test_expired_recent_memory_does_not_touch_long_term_facts(self):
        await self.mem.save_turn(1, 1, "hi", "hello")
        await self.mem.add_fact(1, 1, "user ko cricket pasand hai", source="explicit")
        self.db.memory.docs.clear()                      # what the TTL monitor does after MEMORY_DAYS
        self.assertEqual(await self.mem.load_history(1, 1), [])
        self.assertEqual(len(await self.mem.list_facts(1, 1)), 1)

    async def test_ttl_option_conflict_is_repaired_with_collmod(self):
        self.db.ttl_conflict = True
        self.assertTrue(await self.mem.ensure_indexes())
        self.assertTrue(any(args[0] == "collMod" for args, _ in self.db.commands))


class TestLongTermFacts(MemCase):
    async def test_add_retrieve_only_relevant(self):
        await self.mem.add_fact(1, 1, "user ko cricket aur chess pasand hai")
        await self.mem.add_fact(1, 1, "user python bot banana seekh raha hai")
        got = await self.mem.relevant_facts(1, 1, ["aaj cricket match dekha"])
        self.assertEqual(got, ["user ko cricket aur chess pasand hai"])
        self.assertEqual(await self.mem.relevant_facts(1, 1, ["kya haal hai"]), [])    # nothing relevant -> nothing injected

    async def test_prompt_budget_is_bounded(self):
        for i in range(10):
            await self.mem.add_fact(1, 1, f"user ko cricket topic{i} pasand hai", key=f"k{i}")
        with mock.patch.object(memmod, "LONG_MEMORY_IN_PROMPT", 3):
            self.assertEqual(len(await self.mem.relevant_facts(1, 1, ["cricket"])), 3)
        with mock.patch.object(memmod, "LONG_MEMORY_PROMPT_CHARS", 60):
            self.assertLessEqual(sum(map(len, await self.mem.relevant_facts(1, 1, ["cricket"]))), 60)

    async def test_duplicates_are_avoided(self):
        self.assertEqual(await self.mem.add_fact(1, 1, "user ko chai pasand hai"), "added")
        self.assertEqual(await self.mem.add_fact(1, 1, "User ko chai pasand hai!"), "duplicate")
        self.assertEqual(len(await self.mem.list_facts(1, 1)), 1)

    async def test_same_key_updates_in_place(self):
        await self.mem.add_fact(1, 1, "favourite color blue hai", key="fav_color")
        self.assertEqual(await self.mem.add_fact(1, 1, "favourite color ab green hai", key="fav_color"), "updated")
        facts = await self.mem.list_facts(1, 1)
        self.assertEqual([f["fact"] for f in facts], ["favourite color ab green hai"])

    async def test_isolation_between_users_and_chats(self):
        await self.mem.add_fact(10, 1, "user ko cricket pasand hai")
        self.assertEqual(await self.mem.relevant_facts(10, 2, ["cricket"]), [])     # other user, same chat
        self.assertEqual(await self.mem.relevant_facts(11, 1, ["cricket"]), [])     # same user, other chat
        self.assertEqual(len(await self.mem.relevant_facts(10, 1, ["cricket"])), 1)
        self.assertEqual(await self.mem.list_facts(10, 2), [])

    async def test_delete_by_number_and_clear(self):
        for i, f in enumerate(["user ko chai pasand hai", "user cricket khelta hai", "user guitar seekh raha hai"]):
            await self.mem.add_fact(1, 1, f, key=f"k{i}")
        removed = await self.mem.delete_fact(1, 1, 2)
        self.assertEqual(removed, "user cricket khelta hai")
        self.assertEqual(len(await self.mem.list_facts(1, 1)), 2)
        self.assertIsNone(await self.mem.delete_fact(1, 1, 9))
        await self.mem.add_fact(1, 2, "dusre user ki baat chess hai", key="z")
        self.assertEqual(await self.mem.clear_facts(1, 1), 2)
        self.assertEqual(await self.mem.list_facts(1, 1), [])
        self.assertEqual(len(await self.mem.list_facts(1, 2)), 1)                    # others untouched

    async def test_fact_limit_evicts_oldest_auto_fact_but_keeps_explicit(self):
        with mock.patch.object(memmod, "LONG_MEMORY_MAX_FACTS", 3):
            await self.mem.add_fact(1, 1, "pehla explicit fact chess", key="a", source="explicit")
            await self.mem.add_fact(1, 1, "doosra auto fact cricket", key="b")
            await self.mem.add_fact(1, 1, "teesra auto fact guitar", key="c")
            self.assertEqual(await self.mem.add_fact(1, 1, "chautha auto fact coding", key="d"), "added")
            facts = [f["fact"] for f in await self.mem.list_facts(1, 1)]
        self.assertIn("pehla explicit fact chess", facts)
        self.assertNotIn("doosra auto fact cricket", facts)
        self.assertEqual(len(facts), 3)

    async def test_unique_index_race_is_treated_as_duplicate(self):
        await self.mem.ensure_indexes()
        await self.mem.add_fact(1, 1, "user ko chai pasand hai", key="chai")
        with mock.patch.object(Memory, "list_facts", return_value=[]):      # simulate losing the race
            self.assertEqual(await self.mem.add_fact(1, 1, "user ko chai bahut pasand hai", key="chai"), "duplicate")


class TestSafetyFilter(unittest.TestCase):
    def test_rejects_secrets_contacts_and_sensitive(self):
        bad = [
            "mera password hunter2 hai", "my OTP is 123456", "api key sk-abcdefghijklmnop", "session string BQHN5k4A",
            "phone number 9876543210 hai", "email me at a.b@gmail.com", "mera username @someone_cool hai",
            "check https://example.com/me", "ghar ka address 12 MG road hai", "card number 4111 1111 1111 1111",
            "user depression se guzar raha hai", "user muslim hai", "user ki salary 50k hai", "user ko cancer hai",
            "user 15 years old hai", "user class 9 me padhta hai", "wo sex ke baare me baat karta hai",
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertFalse(is_safe_fact(text)[0], text)

    def test_accepts_normal_preferences(self):
        good = ["user ko cricket pasand hai", "user python seekh raha hai", "user ko Cherry bulana pasand hai",
                "user ek music bot project par kaam kar raha hai", "user ko chai pasand hai coffee nahi"]
        for text in good:
            with self.subTest(text=text):
                self.assertTrue(is_safe_fact(text)[0], text)

    def test_length_limits(self):
        self.assertFalse(is_safe_fact("hi")[0])
        self.assertFalse(is_safe_fact("x " * 200)[0])


class TestExtraction(MemCase):
    def test_parse_valid_fenced_and_partial_json(self):
        good = '{"facts":[{"key":"fav_game","fact":"user ko chess pasand hai","category":"interest"}]}'
        self.assertEqual(parse_extraction(good)[0]["key"], "fav_game")
        self.assertEqual(parse_extraction("```json\n" + good + "\n```")[0]["category"], "interest")
        self.assertEqual(parse_extraction("Sure! " + good + " hope it helps")[0]["fact"], "user ko chess pasand hai")

    def test_parse_malformed_never_raises(self):
        for bad in [None, "", "not json", "{", '{"facts": "oops"}', '{"facts":[1,2]}', '{"facts":[{"key":"a"}]}',
                    '{"facts":[{"fact":123}]}', "[]", '{"facts":null}']:
            with self.subTest(bad=bad):
                self.assertEqual(parse_extraction(bad), [])

    def test_parse_limits_and_bad_category(self):
        many = '{"facts":[' + ",".join('{"fact":"user ko cricket %d pasand hai","category":"weird"}' % i for i in range(9)) + "]}"
        out = parse_extraction(many)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0]["category"], "other")

    async def test_extract_and_store_success_filters_unsafe_and_updates_by_key(self):
        history = [{"role": "user", "content": "mujhe chess pasand hai, mera number 9876543210"}]

        async def ask(messages, max_tokens=None, temperature=None):
            ask.calls.append((messages, max_tokens, temperature))
            return ('{"facts":[{"key":"fav_game","fact":"user ko chess pasand hai"},'
                    '{"key":"phone","fact":"user ka number 9876543210 hai"}]}')
        ask.calls = []
        self.assertEqual(await self.mem.extract_and_store(1, 1, history, ask), 1)           # phone number rejected
        self.assertEqual([f["fact"] for f in await self.mem.list_facts(1, 1)], ["user ko chess pasand hai"])
        self.assertEqual(ask.calls[0][2], 0.2)                                              # deterministic extraction

        async def ask2(messages, max_tokens=None, temperature=None):
            return '{"facts":[{"key":"fav_game","fact":"user ko ab carrom pasand hai"}]}'
        self.assertEqual(await self.mem.extract_and_store(1, 1, history, ask2), 1)
        self.assertEqual([f["fact"] for f in await self.mem.list_facts(1, 1)], ["user ko ab carrom pasand hai"])

    async def test_extract_survives_malformed_json_api_failure_and_exceptions(self):
        history = [{"role": "user", "content": "hello there"}]

        async def garbage(*a, **k): return "I cannot do that"
        async def nothing(*a, **k): return None
        async def boom(*a, **k): raise RuntimeError("api exploded")
        for fn in (garbage, nothing, boom):
            self.assertEqual(await self.mem.extract_and_store(1, 1, history, fn), 0)
        self.assertEqual(await self.mem.list_facts(1, 1), [])

    def test_extraction_cadence_and_explicit_trigger(self):
        with mock.patch.object(memmod, "MEMORY_EXTRACT_EVERY", 3):
            self.assertEqual([self.mem.should_extract(1, 1, "hello") for _ in range(6)], [False, False, True, False, False, True])
            self.assertTrue(self.mem.should_extract(1, 1, "yaad rakhna mujhe chai pasand hai"))
            self.assertTrue(self.mem.should_extract(1, 2, "my name is Aman"))
        with mock.patch.object(memmod, "MEMORY_EXTRACT_EVERY", 0):
            self.assertFalse(any(self.mem.should_extract(2, 2, "hello") for _ in range(20)))

    def test_facts_block_formatting(self):
        self.assertEqual(format_facts_block([]), "")
        block = format_facts_block(["user ko chai pasand hai"])
        self.assertIn("- user ko chai pasand hai", block)


class TestDatabaseOutage(MemCase):
    async def test_every_operation_degrades_gracefully(self):
        self.db.fail = True
        self.assertEqual(await self.mem.load_history(1, 1), [])
        self.assertFalse(await self.mem.save_turn(1, 1, "a", "b"))
        self.assertEqual(await self.mem.list_facts(1, 1), [])
        self.assertEqual(await self.mem.add_fact(1, 1, "user ko chai pasand hai"), "error")
        self.assertIsNone(await self.mem.delete_fact(1, 1, 1))
        self.assertEqual(await self.mem.clear_facts(1, 1), 0)
        self.assertEqual(await self.mem.relevant_facts(1, 1, ["chai"]), [])
        self.assertIsNone(await self.mem.get_enabled(5))
        self.assertFalse(await self.mem.set_enabled(5, False))
        self.assertFalse(await self.mem.ensure_indexes())
        self.assertFalse(self.mem.ok)
        with self.assertRaises(Exception):
            await self.mem.ping()

    async def test_recovers_when_database_returns(self):
        self.db.fail = True
        await self.mem.save_turn(1, 1, "a", "b")
        self.assertFalse(self.mem.ok)
        self.db.fail = False
        self.assertTrue(await self.mem.save_turn(1, 1, "a", "b"))
        self.assertTrue(self.mem.ok)

    async def test_slow_database_times_out(self):
        import asyncio

        class Slow(FakeDB):
            async def command(self, *a, **k): await asyncio.sleep(5)
        mem = Memory(Slow())
        with mock.patch.object(memmod, "DB_OP_TIMEOUT", 0.05):
            with self.assertRaises(asyncio.TimeoutError):
                await mem.ping()
