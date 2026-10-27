"""Translate broker events into Open Responses SSE events."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from aegra_api.core.sse import format_sse_message
from aegra_api.models import Run
from aegra_api.services.responses_mapping import (
    _as_mapping,
    _message_role,
    _text_from_content,
    interrupt_items,
    project_response_output,
)
from aegra_api.services.streaming_service import streaming_service

_TERMINAL_RUN_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})


class ResponsesTranslator:
    """Fold ``messages`` and ``values`` events into Responses SSE payloads."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = dict(response)
        initial = response.get("output")
        self._initial_output: list[dict[str, Any]] = list(initial) if isinstance(initial, list) else []
        self.response["output"] = []
        self._text: dict[str, str] = {}
        self._opened: set[str] = set()
        self._indexes: dict[str, int] = {}
        self._output_items: list[dict[str, Any]] = []
        self._interrupt_ids: set[str] = set()
        self._last_values: dict[str, Any] | None = None
        self._closed = False
        self.saw_interrupt = False

    def opening_events(self) -> list[dict[str, Any]]:
        """``response.created`` and ``response.in_progress`` before broker events."""
        created = dict(self.response)
        created["status"] = "in_progress"
        created["output"] = []
        return [
            {"type": "response.created", "response": created},
            {"type": "response.in_progress", "response": created},
        ]

    def feed(self, event_type: str, payload: Any) -> list[dict[str, Any]]:
        """Translate one broker event. Later events after a terminal event are ignored."""
        if self._closed:
            return []
        if event_type in ("messages/partial", "messages/complete"):
            return self._on_messages(event_type, payload)
        if event_type == "values" or event_type.startswith("values"):
            if isinstance(payload, dict):
                self._last_values = payload
            return self._on_interrupts(payload)
        if event_type == "error":
            return self._fail(_error_message(payload))
        if event_type == "end":
            return self._on_end(payload if isinstance(payload, dict) else {})
        return []

    def snapshot_events(self) -> list[dict[str, Any]]:
        """A finished response as a short SSE sequence, with no token deltas."""
        events = self.opening_events()
        status = self.response.get("status")
        if status == "failed":
            events.extend(self._fail(self._stored_error()))
            return events
        self._copy_output(self._initial_output)
        if status == "incomplete":
            events.append(self._terminal("response.incomplete"))
            return events
        events.append(self._terminal("response.completed"))
        return events

    def _on_messages(self, event_type: str, payload: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for message in _message_payloads(payload):
            mapped = _as_mapping(message)
            if mapped is None or _message_role(mapped) != "assistant":
                continue
            raw_id = mapped.get("id")
            item_id = raw_id if isinstance(raw_id, str) else "msg"
            text = _text_from_content(mapped.get("content"))
            events.extend(self._ensure_open(item_id))
            previous = self._text.get(item_id, "")
            if text.startswith(previous):
                delta = text[len(previous) :]
            else:
                delta = text
            self._text[item_id] = text
            if delta:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": self._indexes[item_id],
                        "content_index": 0,
                        "delta": delta,
                    }
                )
            if event_type == "messages/complete":
                events.extend(self._close_message(item_id, text))
        return events

    def _ensure_open(self, item_id: str) -> list[dict[str, Any]]:
        if item_id in self._opened:
            return []
        index = len(self._output_items)
        self._indexes[item_id] = index
        self._opened.add(item_id)
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        self._output_items.append(item)
        return [
            {"type": "response.output_item.added", "output_index": index, "item": item},
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]

    def _close_message(self, item_id: str, text: str) -> list[dict[str, Any]]:
        index = self._indexes[item_id]
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        self._output_items[index] = item
        return [
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "text": text,
            },
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]

    def _on_interrupts(self, payload: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for item in interrupt_items(payload if isinstance(payload, dict) else None):
            interrupt_id = item["id"]
            if interrupt_id in self._interrupt_ids:
                continue
            self._interrupt_ids.add(interrupt_id)
            self.saw_interrupt = True
            index = len(self._output_items)
            self._output_items.append(item)
            events.append({"type": "response.output_item.added", "output_index": index, "item": item})
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        return events

    def _on_end(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        status = payload.get("status")
        if status == "error":
            return self._fail("execution failed")
        if status == "interrupted" and self.saw_interrupt:
            self.response["status"] = "incomplete"
            self.response["incomplete_details"] = {"reason": "approval_required"}
            self._fill_from_values_if_empty()
            return [self._terminal("response.incomplete")]
        if status == "interrupted":
            self.response["status"] = "cancelled"
            self._fill_from_values_if_empty()
            return [self._terminal("response.completed")]
        self.response["status"] = "completed"
        self._fill_from_values_if_empty()
        return [self._terminal("response.completed")]

    def _fail(self, message: str) -> list[dict[str, Any]]:
        self.response["status"] = "failed"
        self.response["error"] = {"code": "server_error", "message": message}
        return [self._terminal("response.failed")]

    def _fill_from_values_if_empty(self) -> None:
        """Use the last values snapshot when no assistant message was streamed."""
        has_message = any(item.get("type") == "message" for item in self._output_items)
        if has_message or not isinstance(self._last_values, dict):
            return
        projected = project_response_output(self._last_values)
        if projected:
            self._copy_output(projected)

    def _copy_output(self, items: list[dict[str, Any]]) -> None:
        self._output_items = items
        self.response["output"] = items

    def _terminal(self, event_type: str) -> dict[str, Any]:
        self._closed = True
        self.response["output"] = list(self._output_items)
        return {"type": event_type, "response": dict(self.response)}

    def _stored_error(self) -> str:
        error = self.response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        return "Run failed"


def encode_responses_event(event: dict[str, Any], event_id: str | None) -> str:
    """Serialize one Responses event as an SSE frame."""
    event_type = event.get("type")
    name = event_type if isinstance(event_type, str) else "response.completed"
    return format_sse_message(name, event, event_id)


async def iter_responses_sse(
    run: Run,
    response: dict[str, Any],
    *,
    starting_after: str | None,
) -> AsyncGenerator[str, None]:
    """Stream a run, replaying broker history so deltas stay aligned across reconnects."""
    translator = ResponsesTranslator(response)
    if run.status in _TERMINAL_RUN_STATUSES and not streaming_service.is_run_streaming(run.run_id):
        for event in translator.snapshot_events():
            yield encode_responses_event(event, None)
        return

    past = starting_after is None
    if past:
        for event in translator.opening_events():
            yield encode_responses_event(event, None)

    async for event_id, event_type, payload in streaming_service.iter_broker_events(run, None):
        produced = translator.feed(event_type, payload)
        if not past:
            if event_id == starting_after:
                past = True
            continue
        for event in produced:
            yield encode_responses_event(event, event_id)


def _message_payloads(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if payload is None:
        return []
    return [payload]


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"]
    return "execution failed"
