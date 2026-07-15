"""Unit tests for the signed webhook sender + durable outbox routing.

No real HTTP or Supabase — httpx and the outbox repo are mocked.
"""

import hashlib
import hmac
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx

from voice_engine.platform import webhooks as webhooks_mod
from voice_engine.platform.webhooks import WebhookSender

SECRET = "shared-secret"


def _patch_httpx(monkeypatch, *, status_code: int = 200):
    """Patch httpx.AsyncClient; return a captured-calls dict."""
    captured: dict = {}

    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = "" if status_code < 400 else "err"
    if status_code >= 400:
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=response
            )
        )
    else:
        response.raise_for_status = MagicMock()

    async def _post(url, content=None, headers=None):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        return response

    @asynccontextmanager
    async def _client(*args, **kwargs):
        client = MagicMock()
        client.post = _post
        yield client

    monkeypatch.setattr(webhooks_mod.httpx, "AsyncClient", _client)
    return captured


def _make_sender() -> WebhookSender:
    sender = WebhookSender(callback_url="https://smrtesy.test/api/voice/webhook",
                           callback_secret=SECRET)
    # Isolate from the DB: replace the outbox with an async mock.
    sender.outbox = MagicMock()
    sender.outbox.enqueue = AsyncMock(return_value="outbox-1")
    sender.outbox.mark_delivered = AsyncMock()
    sender.outbox.record_failure = AsyncMock()
    return sender


async def test_deliver_signs_exact_bytes(monkeypatch):
    captured = _patch_httpx(monkeypatch)
    sender = _make_sender()

    payload = '{"event_type":"smrtvoice.job.started","data":{}}'
    ok = await sender.deliver(payload)

    assert ok is True
    # The exact payload string is what was POSTed (not a re-serialization).
    assert captured["content"] == payload
    ts = captured["headers"]["X-Webhook-Timestamp"]
    expected = hmac.new(
        SECRET.encode(), f"{ts}.{payload}".encode(), hashlib.sha256
    ).hexdigest()
    # smrtesy verifies HMAC-SHA256 over `${timestamp}.${rawBody}` with the same
    # shared secret — this is the exact string it will reconstruct.
    assert captured["headers"]["X-Webhook-Signature"] == expected


async def test_job_started_is_durable(monkeypatch):
    _patch_httpx(monkeypatch)
    sender = _make_sender()

    ok = await sender.send_job_started(uuid4(), uuid4(), uuid4())

    assert ok is True
    sender.outbox.enqueue.assert_awaited_once()
    assert sender.outbox.enqueue.await_args.kwargs["event_type"] == "smrtvoice.job.started"
    sender.outbox.mark_delivered.assert_awaited_once_with("outbox-1")


async def test_line_completed_is_not_durable(monkeypatch):
    _patch_httpx(monkeypatch)
    sender = _make_sender()

    ok = await sender.send_line_completed(uuid4(), uuid4(), uuid4(), {"line_number": 1})

    assert ok is True
    # High-volume, best-effort, already has a direct DB path — never enqueued.
    sender.outbox.enqueue.assert_not_awaited()


async def test_failed_delivery_leaves_row_pending(monkeypatch):
    _patch_httpx(monkeypatch, status_code=404)
    sender = _make_sender()
    # Skip the 5x tenacity backoff waits in the test.
    sender._deliver_with_retry = AsyncMock(side_effect=RuntimeError("404"))

    ok = await sender.send_job_completed(
        uuid4(), uuid4(), uuid4(),
        result=MagicMock(model_dump=MagicMock(return_value={})),
    )

    # Non-fatal: caller keeps going, but the outbox row is recorded for retry.
    assert ok is False
    sender.outbox.enqueue.assert_awaited_once()
    sender.outbox.mark_delivered.assert_not_awaited()
    sender.outbox.record_failure.assert_awaited_once()
