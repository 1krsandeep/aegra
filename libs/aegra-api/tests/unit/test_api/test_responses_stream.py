"""Unit tests for translating broker events into Responses SSE."""

from typing import Any

from aegra_api.services.responses_stream import ResponsesTranslator, encode_responses_event


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


def test_encode_uses_the_event_type_as_the_sse_name() -> None:
    """Clients dispatch on the SSE event name, which matches the payload type."""
    frame = encode_responses_event({"type": "response.output_text.delta", "delta": "Hi"}, "run-1_event_2")

    assert frame.startswith("event: response.output_text.delta\n")
    assert "id: run-1_event_2" in frame
    assert '"delta":"Hi"' in frame
