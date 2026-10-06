"""Provider layer: DeepSeek-correct payloads, error handling, retry/backoff, cooldown, fallback."""
import asyncio
import logging
import time
import unittest

import providers
from providers import Provider, ask_deepseek, parse_retry_after
from tests.fakes import FakeHTTP, FakeResponse, ok, err

DS = "https://api.deepseek.com/chat/completions"
GQ = "https://api.groq.com/openai/v1/chat/completions"
MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


class ProviderCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.http = FakeHTTP()
        providers.set_http(self.http)
        self.sleeps = []

        async def fake_sleep(s): self.sleeps.append(s)
        self._old_sleep, providers._sleep = providers._sleep, fake_sleep
        for p in providers.PROVIDERS:
            p.cooldown_until, p.strip_extras = 0.0, False
        self.ds, self.gq = providers.PROVIDERS[0], providers.PROVIDERS[1]

    async def asyncTearDown(self):
        providers._sleep = self._old_sleep


class TestPayloads(ProviderCase):
    async def test_deepseek_payload_follows_official_docs(self):
        await ask_deepseek(MSG)
        body = self.http.calls(DS)[0]["json"]
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})   # thinking is ON by default at DeepSeek
        self.assertNotIn("frequency_penalty", body)                # deprecated at DeepSeek: no effect
        self.assertNotIn("presence_penalty", body)
        self.assertNotIn("top_p", body)                            # ignored outside thinking mode
        self.assertIn("temperature", body)
        self.assertEqual(self.http.calls(DS)[0]["headers"]["Authorization"], f"Bearer {self.ds.key}")

    async def test_groq_payload(self):
        self.ds.cool_down(999)
        await ask_deepseek(MSG)
        body = self.http.calls(GQ)[0]["json"]
        self.assertIn("top_p", body)
        self.assertIn("frequency_penalty", body)
        self.assertEqual(body["reasoning_effort"], "low")          # gpt-oss on Groq

    async def test_user_extra_always_sent_and_overrides(self):
        p = Provider("x", "https://example.com/v1/chat/completions", "k", "m", {"reasoning_effort": "none", "top_p": 0.5})
        payload = p.build_payload(MSG, 50, 1.0)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["top_p"], 0.5)

    async def test_unsupported_optional_params_are_dropped_and_remembered(self):
        self.ds.cool_down(999)
        self.http.queue(GQ, err(400, '{"error":{"message":"unsupported parameter: reasoning_effort"}}'), ok("hmm"))
        self.assertEqual(await ask_deepseek(MSG), "hmm")
        first, second = [c["json"] for c in self.http.calls(GQ)]
        self.assertIn("reasoning_effort", first)
        self.assertNotIn("reasoning_effort", second)
        self.assertTrue(self.gq.strip_extras)
        await ask_deepseek(MSG)                                    # remembered for next calls
        self.assertNotIn("reasoning_effort", self.http.calls(GQ)[-1]["json"])


class TestErrors(ProviderCase):
    async def test_429_fails_over_without_retrying_and_is_skipped_afterwards(self):
        self.http.queue(DS, err(429, '{"error":"too fast"}'))
        self.http.queue(GQ, ok("backup answer"))
        self.assertEqual(await ask_deepseek(MSG), "backup answer")
        self.assertEqual(len(self.http.calls(DS)), 1)              # no hammering
        self.assertFalse(self.ds.available())
        self.http.queue(GQ, ok("again"))
        await ask_deepseek(MSG)
        self.assertEqual(len(self.http.calls(DS)), 1)              # skipped while cooling down

    async def test_daily_limit_gets_long_cooldown(self):
        self.http.queue(DS, err(429, '{"error":{"code":"free_trial_rate_limit_exceeded","message":"limited to 100 requests per day"}}'))
        await ask_deepseek(MSG)
        self.assertGreater(self.ds.cooldown_until - time.monotonic(), 1000)

    async def test_retry_after_header_is_respected(self):
        self.http.queue(DS, err(429, "slow down", {"Retry-After": "120"}))
        await ask_deepseek(MSG)
        left = self.ds.cooldown_until - time.monotonic()
        self.assertTrue(100 < left <= 121, left)

    def test_retry_after_parsing(self):
        self.assertEqual(parse_retry_after("7"), 7.0)
        self.assertIsNone(parse_retry_after("garbage"))
        self.assertIsNone(parse_retry_after(None))

    async def test_timeout_retries_with_backoff_then_fails_over(self):
        self.http.queue(DS, *[asyncio.TimeoutError()] * 5)
        self.http.queue(GQ, ok("saved by backup"))
        self.assertEqual(await ask_deepseek(MSG), "saved by backup")
        self.assertEqual(len(self.http.calls(DS)), 1 + providers.PROVIDER_RETRIES)
        self.assertEqual(len(self.sleeps), providers.PROVIDER_RETRIES)   # exponential backoff between tries
        self.assertFalse(self.ds.available())

    async def test_transient_503_retried_on_same_provider(self):
        self.http.queue(DS, err(503, "overloaded"), ok("fine now"))
        self.assertEqual(await ask_deepseek(MSG), "fine now")
        self.assertEqual(len(self.http.calls(DS)), 2)
        self.assertEqual(len(self.http.calls(GQ)), 0)

    async def test_500_after_all_retries_fails_over(self):
        self.http.queue(DS, *[err(500, "boom")] * 5)
        self.http.queue(GQ, ok("backup"))
        self.assertEqual(await ask_deepseek(MSG), "backup")

    async def test_auth_balance_and_model_errors_are_not_retried(self):
        for status in (401, 402, 404):
            with self.subTest(status=status):
                for p in providers.PROVIDERS:
                    p.cooldown_until = 0.0
                    self.http.requests.clear()
                self.http.queue(DS, err(status, "x"))
                self.http.queue(GQ, ok("backup"))
                self.assertEqual(await ask_deepseek(MSG), "backup")
                self.assertEqual(len(self.http.calls(DS)), 1)
                self.assertGreater(self.ds.cooldown_until - time.monotonic(), 1000)

    async def test_400_422_without_optional_param_fails_over_with_short_cooldown(self):
        for status in (400, 422):
            with self.subTest(status=status):
                for p in providers.PROVIDERS:
                    p.cooldown_until, p.strip_extras = 0.0, False
                self.http.requests.clear()
                self.http.queue(DS, err(status, '{"error":"messages must not be empty"}'))
                self.http.queue(GQ, ok("backup"))
                self.assertEqual(await ask_deepseek(MSG), "backup")
                self.assertEqual(len(self.http.calls(DS)), 1)
                self.assertLess(self.ds.cooldown_until - time.monotonic(), 60)

    async def test_deepseek_400_mentioning_thinking_retries_without_it(self):
        self.http.queue(DS, err(422, '{"error":"unknown parameter: thinking"}'), ok("plain"))
        self.assertEqual(await ask_deepseek(MSG), "plain")
        self.assertNotIn("thinking", self.http.calls(DS)[1]["json"])

    async def test_all_providers_failing_returns_none(self):
        self.http.queue(DS, err(500, "x"), err(500, "x"), err(500, "x"))
        self.http.queue(GQ, err(401, "x"))
        self.assertIsNone(await ask_deepseek(MSG))

    async def test_all_providers_cooling_down_makes_no_request(self):
        for p in providers.PROVIDERS:
            p.cool_down(100)
        self.assertIsNone(await ask_deepseek(MSG))
        self.assertEqual(self.http.requests, [])


class TestBadResponses(ProviderCase):
    async def test_empty_reply_retried_once_with_triple_tokens(self):
        self.http.queue(DS, ok("", "length"), ok("ab bolo"))
        self.assertEqual(await ask_deepseek(MSG, max_tokens=100), "ab bolo")
        self.assertEqual([c["json"]["max_tokens"] for c in self.http.calls(DS)], [100, 300])

    async def test_null_content_is_treated_as_empty(self):
        self.http.queue(DS, FakeResponse(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}),
                        FakeResponse(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}))
        self.http.queue(GQ, ok("backup"))
        self.assertEqual(await ask_deepseek(MSG), "backup")

    async def test_content_filter_goes_to_next_provider_without_retry(self):
        self.http.queue(DS, ok("", "content_filter"))
        self.http.queue(GQ, ok("safe answer"))
        self.assertEqual(await ask_deepseek(MSG), "safe answer")
        self.assertEqual(len(self.http.calls(DS)), 1)

    async def test_malformed_json_and_missing_choices_do_not_crash(self):
        self.http.queue(DS, FakeResponse(200, bad_json=True))
        self.http.queue(GQ, FakeResponse(200, {"unexpected": True}))
        self.assertIsNone(await ask_deepseek(MSG))

    async def test_content_parts_list_is_supported(self):
        self.http.queue(DS, FakeResponse(200, {"choices": [{"message": {"content": [{"type": "text", "text": "hello"}]}, "finish_reason": "stop"}]}))
        self.assertEqual(await ask_deepseek(MSG), "hello")

    async def test_reply_is_cleaned(self):
        self.http.queue(DS, ok('Riya: "[sticker: 😂] arre wah"'))
        self.assertEqual(await ask_deepseek(MSG), "arre wah")


class TestNoSecretsInLogs(ProviderCase):
    async def test_keys_never_appear_in_log_output(self):
        self.http.queue(DS, err(401, "bad key sk-deepseekkey0123456789"), )
        self.http.queue(GQ, err(500, "x"), err(500, "x"), err(500, "x"))
        with self.assertLogs("girl-chatbot.providers", level=logging.DEBUG) as cm:
            await ask_deepseek(MSG)
        text = "\n".join(cm.output)
        # the provider layer never prints headers/payloads; the API error body is truncated and goes through
        # the redacting formatter (see test_infra) before reaching real logs
        self.assertNotIn(self.gq.key, text)
        self.assertNotIn("Authorization", text)
