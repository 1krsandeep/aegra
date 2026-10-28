"""Unit tests for translating broker events into Responses SSE."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from aegra_api.models import Run
from aegra_api.services.responses_stream import ResponsesTranslator, encode_responses_event, iter_responses_sse


def _response() -> dict[str, Any]:
    return {
        "id": "run-1",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": "agent",
        "output": [],
        "previous_response_id": None,
        "conversation": None,
        "metadata": {},
        "background": False,
    }


def test_partial_messages_emit_deltas_and_completion() -> None:
    """Token snapshots become output_text deltas, then a completed response."""
    translator = ResponsesTranslator(_response())
    opening = [event["type"] for event in translator.opening_events()]
    partial = translator.feed(
        "messages/partial",
        [{"id": "m1", "type": "ai", "content": "Hi"}],
    )
    more = translator.feed(
        "messages/partial",
        [{"id": "m1", "type": "ai", "content": "Hi there"}],
    )
    done = translator.feed(
        "messages/complete",
        [{"id": "m1", "type": "ai", "content": "Hi there"}],
    )
    finished = translator.feed("end", {"status": "success"})

    assert opening == ["response.created", "response.in_progress"]
    assert [event["type"] for event in partial] == [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
    ]
    assert partial[-1]["delta"] == "Hi"
    assert more[0]["delta"] == " there"
    assert [event["type"] for event in done] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]
    assert finished[0]["type"] == "response.completed"
    assert finished[0]["response"]["status"] == "completed"
    assert finished[0]["response"]["output"][0]["content"][0]["text"] == "Hi there"


def test_interrupt_end_is_an_incomplete_response() -> None:
    """A values interrupt followed by an interrupted end asks the client to approve."""
    translator = ResponsesTranslator(_response())
    added = translator.feed(
        "values",
        {"__interrupt__": [{"id": "int-1", "value": {"action_request": {"action": "search"}}}]},
    )
    finished = translator.feed("end", {"status": "interrupted"})

    assert added[0]["item"]["type"] == "mcp_approval_request"
    assert added[0]["item"]["id"] == "int-1"
    assert finished[0]["type"] == "response.incomplete"
    assert finished[0]["response"]["status"] == "incomplete"


def test_cancel_end_without_an_interrupt_completes_as_cancelled() -> None:
    """An interrupted end with no interrupt payload is a cancel, not an approval."""
    translator = ResponsesTranslator(_response())
    finished = translator.feed("end", {"status": "interrupted"})

    assert finished[0]["type"] == "response.completed"
    assert finished[0]["response"]["status"] == "cancelled"


def test_values_only_success_projects_the_final_message() -> None:
    """Graphs that emit values without message deltas still fill the completed output."""
    translator = ResponsesTranslator(_response())
    translator.feed("values", {"messages": [{"id": "a1", "type": "ai", "content": "Done"}]})
    finished = translator.feed("end", {"status": "success"})

    assert finished[0]["response"]["output"][0]["content"][0]["text"] == "Done"


def test_snapshot_of_a_finished_response_includes_its_output() -> None:
    """Reattaching after the broker is gone replays the stored output, not an empty body."""
    stored = _response()
    stored["status"] = "completed"
    stored["output"] = [
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Done", "annotations": []}],
        }
    ]
    events = ResponsesTranslator(stored).snapshot_events()

    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Done"


def test_end_closes_a_partial_message_and_keeps_tool_calls() -> None:
    """A partial without a complete event still finishes from the final values, including tools."""
    translator = ResponsesTranslator(_response())
    translator.feed("messages/partial", [{"id": "m1", "type": "ai", "content": ""}])
    translator.feed(
        "values",
        {
            "messages": [
                {
                    "id": "m1",
                    "type": "ai",
                    "content": "Hello",
                    "tool_calls": [{"id": "call-1", "name": "search", "args": {}}],
                }
            ]
        },
    )
    finished = translator.feed("end", {"status": "success"})
    output = finished[0]["response"]["output"]

    assert output[0]["status"] == "completed"
    assert output[0]["content"][0]["text"] == "Hello"
    assert output[1]["type"] == "function_call"
    assert output[1]["name"] == "search"


def test_native_message_events_become_text_deltas() -> None:
    """Agent Protocol v2 message events are not dropped on this surface."""
    translator = ResponsesTranslator(_response())
    start = translator.feed("messages", {"event": "message-start", "role": "ai", "id": "m1"})
    delta = translator.feed(
        "messages",
        {"event": "content-block-delta", "delta": {"type": "text-delta", "text": "Hi"}},
    )
    done = translator.feed("messages", {"event": "message-finish"})
    finished = translator.feed("end", {"status": "success"})

    assert start[0]["type"] == "response.output_item.added"
    assert delta[-1]["type"] == "response.output_text.delta"
    assert delta[-1]["delta"] == "Hi"
    assert done[0]["type"] == "response.output_text.done"
    assert finished[0]["response"]["output"][0]["content"][0]["text"] == "Hi"


def _run(status: str) -> Run:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    return Run(
        run_id="run-1",
        thread_id="thread-1",
        assistant_id="assistant-1",
        status=status,
        user_id="test-user",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_missing_cursor_with_history_emits_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cursor that is not in the buffer must not swallow the rest of the stream."""

    async def fake_iter(_run: Run, _last_event_id: str | None = None) -> Any:
        yield ("run-1_event_4", "end", {"status": "success"})

    monkeypatch.setattr(
        "aegra_api.services.responses_stream.streaming_service.iter_broker_events",
        fake_iter,
    )
    frames = [
        frame
        async for frame in iter_responses_sse(
            _run("success"),
            _response(),
            starting_after="not-a-broker-id",
        )
    ]

    assert any("starting_after is no longer available" in frame for frame in frames)


@pytest.mark.asyncio
async def test_evicted_cursor_resumes_from_later_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the named event aged out, later broker events are still delivered."""

    async def fake_iter(_run: Run, _last_event_id: str | None = None) -> Any:
        yield ("run-1_event_8", "end", {"status": "success"})

    monkeypatch.setattr(
        "aegra_api.services.responses_stream.streaming_service.iter_broker_events",
        fake_iter,
    )
    frames = [
        frame
        async for frame in iter_responses_sse(
            _run("success"),
            _response(),
            starting_after="run-1_event_3",
        )
    ]

    assert any("response.completed" in frame for frame in frames)
    assert not any("no longer available" in frame for frame in frames)


def test_native_deltas_stop_when_the_message_closes() -> None:
    """A delta after message-finish, or during a tool message, must not extend closed text."""
    translator = ResponsesTranslator(_response())
    translator.feed("messages", {"event": "message-start", "id": "m1", "role": "assistant"})
    translator.feed(
        "messages",
        {"event": "content-block-delta", "delta": {"type": "text-delta", "text": "Hi"}},
    )
    translator.feed("messages", {"event": "message-finish", "id": "m1"})
    after_close = translator.feed(
        "messages",
        {"event": "content-block-delta", "delta": {"type": "text-delta", "text": " again"}},
    )
    translator.feed("messages", {"event": "message-start", "id": "m2", "role": "assistant"})
    translator.feed("messages", {"event": "message-start", "id": "tool-1", "role": "tool"})
    during_tool = translator.feed(
        "messages",
        {"event": "content-block-delta", "delta": {"type": "text-delta", "text": "nope"}},
    )

    texts = [
        item["content"][0]["text"]
        for item in translator._output_items
        if item.get("type") == "message" and item.get("content")
    ]
    assert after_close == []
    assert during_tool == []
    assert texts[0] == "Hi"
    assert "again" not in "".join(texts)
    assert "nope" not in "".join(texts)


@pytest.mark.asyncio
async def test_finished_response_without_a_broker_snapshots() -> None:
    """A terminal run whose broker was evicted returns the saved response instead of hanging."""
    run = _run("success")
    run.run_id = "run-finished-no-broker"
    saved = _response()
    saved["status"] = "completed"
    saved["output"] = [
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
        }
    ]

    frames: list[str] = []
    async with asyncio.timeout(2):
        async for frame in iter_responses_sse(run, saved, starting_after=None):
            frames.append(frame)

    assert any("response.completed" in frame for frame in frames)
    assert any("Hi" in frame for frame in frames)


def test_closed_translator_ignores_later_events() -> None:
    """A terminal event closes the translator, including a non-object end payload."""
    translator = ResponsesTranslator(_response())
    assert translator.feed("debug", {"ignored": True}) == []
    translator.feed("end", "done")
    assert translator.feed("messages/partial", [{"id": "m1", "type": "ai", "content": "Hi"}]) == []


def test_error_and_failed_snapshot_use_the_stored_message() -> None:
    """Broker errors and a failed snapshot both become response.failed."""
    translator = ResponsesTranslator(_response())
    failed = translator.feed("error", {"message": "boom"})
    assert failed[0]["type"] == "response.failed"
    assert failed[0]["response"]["error"]["message"] == "boom"

    bare = ResponsesTranslator(_response())
    bare.feed("error", None)
    assert bare.response["error"]["message"] == "execution failed"

    stored = _response()
    stored["status"] = "failed"
    stored["error"] = {"message": "saved"}
    snapshot = ResponsesTranslator(stored).snapshot_events()
    assert snapshot[-1]["type"] == "response.failed"
    assert snapshot[-1]["response"]["error"]["message"] == "saved"

    unnamed = _response()
    unnamed["status"] = "failed"
    unnamed["error"] = "not-an-object"
    assert ResponsesTranslator(unnamed).snapshot_events()[-1]["response"]["error"]["message"] == "Run failed"


def test_incomplete_snapshot_and_duplicate_interrupts() -> None:
    """An incomplete snapshot stays incomplete, and the same interrupt is emitted once."""
    stored = _response()
    stored["status"] = "incomplete"
    snapshot = ResponsesTranslator(stored).snapshot_events()
    assert snapshot[-1]["type"] == "response.incomplete"

    translator = ResponsesTranslator(_response())
    payload = {"__interrupt__": [{"id": "int-1", "value": {"name": "search"}}]}
    first = translator.feed("values", payload)
    second = translator.feed("values|inner", payload)
    assert first[0]["item"]["name"] == "search"
    assert second == []
    assert translator.feed("values", ["not-a-dict"]) == []


def test_non_assistant_and_replaced_text_are_handled() -> None:
    """User messages are skipped, and a non-prefix snapshot replaces the accumulated text."""
    translator = ResponsesTranslator(_response())
    assert translator.feed("messages/partial", [{"type": "human", "content": "Hi"}]) == []
    assert translator.feed("messages/partial", None) == []
    opened = translator.feed("messages/partial", {"id": "m1", "type": "ai", "content": "Hi"})
    assert opened[-1]["delta"] == "Hi"
    replaced = translator.feed("messages/partial", {"id": "m1", "type": "ai", "content": "Yo"})
    assert replaced[-1]["delta"] == "Yo"


def test_native_message_shapes_that_do_not_open_an_item() -> None:
    """Finish, unknown events, and wrapped protocol payloads stay aligned with the active item."""
    translator = ResponsesTranslator(_response())
    assert translator.feed("messages", "nope") == []
    assert translator.feed("messages", {"event": "message-finish"}) == []
    assert translator.feed("messages", {"event": "content-block-start"}) == []
    wrapped = translator.feed(
        "messages",
        {"params": {"data": {"event": "message-start", "id": "m9", "role": "assistant"}}},
    )
    assert wrapped[0]["type"] == "response.output_item.added"
    assert translator.feed("messages", {"params": "nope"}) == []


def test_end_error_reconciles_tool_calls_from_final_values() -> None:
    """A failed end closes the response, and a successful end keeps tool calls from values."""
    translator = ResponsesTranslator(_response())
    assert translator.feed("end", {"status": "error"})[0]["type"] == "response.failed"

    with_tools = ResponsesTranslator(_response())
    with_tools.feed("messages/partial", [{"id": "m1", "type": "ai", "content": "Hi"}])
    with_tools.feed(
        "values",
        {
            "messages": [
                {
                    "id": "m1",
                    "type": "ai",
                    "content": "Hi",
                    "tool_calls": [{"name": "search", "id": "call-1", "args": {"q": "x"}}],
                }
            ]
        },
    )
    done = with_tools.feed("end", {"status": "success"})
    output = done[0]["response"]["output"]
    assert any(item.get("type") == "function_call" for item in output)

    replaced = ResponsesTranslator(_response())
    replaced.feed("messages/partial", [{"id": "m1", "type": "ai", "content": "Hi"}])
    replaced._output_items[0]["status"] = "completed"
    replaced._output_items[0]["content"] = "not-a-list"
    replaced.feed("values", {"messages": [{"id": "m1", "type": "ai", "content": "Hi there"}]})
    finished = replaced.feed("end", {"status": "success"})
    assert finished[0]["response"]["output"][0]["content"][0]["text"] == "Hi there"


@pytest.mark.asyncio
async def test_exact_cursor_and_missing_history_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact cursor skips that event, and a terminal run with no history returns the snapshot."""

    async def fake_iter(_run: Run, _last_event_id: str | None = None) -> Any:
        yield ("run-1_event_2", "end", {"status": "success"})

    monkeypatch.setattr(
        "aegra_api.services.responses_stream.streaming_service.iter_broker_events",
        fake_iter,
    )
    frames = [frame async for frame in iter_responses_sse(_run("success"), _response(), starting_after="run-1_event_2")]
    assert frames == []


@pytest.mark.asyncio
async def test_terminal_run_without_history_snapshots_after_a_cursor() -> None:
    """A cursor on a finished run with no broker falls back to the saved response."""
    empty_run = _run("success")
    empty_run.run_id = "run-snapshot-cursor"
    saved = _response()
    saved["status"] = "completed"
    async with asyncio.timeout(2):
        snapshot = [
            frame
            async for frame in iter_responses_sse(
                empty_run,
                saved,
                starting_after="run-snapshot-cursor_event_1",
            )
        ]
    assert any("response.completed" in frame for frame in snapshot)


def test_encode_uses_the_event_type_as_the_sse_name() -> None:
    """Clients dispatch on the SSE event name, which matches the payload type."""
    frame = encode_responses_event({"type": "response.output_text.delta", "delta": "Hi"}, "run-1_event_2")

    assert frame.startswith("event: response.output_text.delta\n")
    assert "id: run-1_event_2" in frame
    assert '"delta":"Hi"' in frame
