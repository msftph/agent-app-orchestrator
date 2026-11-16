"""AG-UI protocol projection over native DTS execution, never local agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Literal

from ag_ui.core import (
    BaseEvent, CustomEvent, RunErrorEvent, RunFinishedEvent, RunFinishedInterruptOutcome, RunStartedEvent,
    TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent,
)
from agent_framework import AgentResponse
from agent_framework_durabletask import (
    AgentSessionId, DurableAgentSession, DurableAIAgentOrchestrationContext,
    DurableWorkflowClient, wrap_workflow_input,
)
from durabletask import task
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.internal.orchestrator_service_pb2 import OrchestrationIdReusePolicy
import grpc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import AGENT_NAME, WORKFLOW_NAME, ResearchReviewResult, ResearchReviewStart, ReviewDecision
from .presentation import response_projection

TURN_ORCHESTRATOR = "ag_ui_conversation_turn"
_INPUT_HASH = "ag_ui_input_sha256"


class GatewayError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=8000, pattern=r"\S")


class ResumeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interrupt_id: str = Field(alias="interruptId", min_length=1, max_length=512)
    status: Literal["resolved"]
    payload: ReviewDecision


class GatewayRequest(BaseModel):
    """A deliberately bounded, trusted-BFF subset of RunAgentInput."""

    model_config = ConfigDict(extra="forbid")
    thread_id: str = Field(alias="threadId", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str = Field(alias="runId", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    parent_run_id: str | None = Field(default=None, alias="parentRunId", max_length=128)
    messages: list[UserMessage] = Field(default_factory=list, max_length=1)
    resume: list[ResumeDecision] = Field(default_factory=list, max_length=1)
    state: dict[str, Any] = Field(default_factory=dict)
    tools: list[Any] = Field(default_factory=list)
    context: list[Any] = Field(default_factory=list)
    forwarded_props: dict[str, Any] = Field(default_factory=dict, alias="forwardedProps")

    @model_validator(mode="after")
    def validate_delta(self) -> GatewayRequest:
        if self.state or self.tools or self.context or self.forwarded_props:
            raise ValueError("Client tools, state, context and forwarded properties are unsupported.")
        if self.messages and self.resume:
            raise ValueError("Send a new user message or a human decision, not both.")
        return self


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scoped_id(caller: str, thread_id: str, run_id: str = "") -> str:
    return "agui-" + fingerprint([caller, thread_id, run_id])


def ag_ui_conversation_turn(
    context: task.OrchestrationContext, payload: dict[str, str],
) -> Generator[task.Task[Any], Any, dict[str, Any]]:
    """Persist each turn's result in DTS, independently of HTTP/receipt lifetimes."""
    agent = DurableAIAgentOrchestrationContext(context, entity_version=2).get_agent(AGENT_NAME)
    session = DurableAgentSession(
        durable_session_id=AgentSessionId(
            name=AGENT_NAME, key=payload["session_key"], entity_version=2,
        ),
    )
    context.set_custom_status({"state": "running"})
    response = yield agent.run(payload["text"], session=session)
    return response.to_dict()


class GatewaySchedulerClient(DurableTaskSchedulerClient):
    def schedule_new_orchestration(self, orchestrator: Any, **kwargs: Any) -> str:
        if not kwargs.get("instance_id"):
            raise ValueError("AG-UI starts require a scoped deterministic instance ID.")
        # Empty replaceableStatus explicitly disallows replacement in every state.
        kwargs["reuse_id_policy"] = OrchestrationIdReusePolicy()
        kwargs["tags"] = {
            **(kwargs.get("tags") or {}), _INPUT_HASH: fingerprint(kwargs.get("input")),
        }
        return super().schedule_new_orchestration(orchestrator, **kwargs)


@dataclass(frozen=True)
class _RpcDetails(grpc.ClientCallDetails):
    method: Any
    timeout: float
    metadata: Any
    credentials: Any
    wait_for_ready: Any
    compression: Any


class GatewayRpcDeadline(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation: Any, details: Any, request: Any) -> Any:
        return continuation(_RpcDetails(
            details.method, min(details.timeout or 15, 15), details.metadata,
            details.credentials, details.wait_for_ready, details.compression,
        ), request)


class DurableGateway:
    def __init__(
        self, client: DurableTaskSchedulerClient, *, mode: str,
        poll_interval: float = 1, timeout: float = 180,
    ):
        self.client = client
        self.mode = mode
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.workflows = DurableWorkflowClient(client, workflow_name=WORKFLOW_NAME, entity_version=2)

    def _name(self, workflow: bool) -> str:
        if workflow:
            # Resolve the official versioned registration name without duplicating it.
            from agent_framework_durabletask import workflow_orchestrator_name
            return workflow_orchestrator_name(WORKFLOW_NAME, entity_version=2)
        return TURN_ORCHESTRATOR

    def _verify(self, state: Any, *, workflow: bool, input_hash: str | None = None) -> None:
        if state.name != self._name(workflow):
            raise GatewayError(404, "AG-UI execution not found.")
        if input_hash is not None and (state.tags or {}).get(_INPUT_HASH) != input_hash:
            raise GatewayError(409, "This thread/run already has different input; use a new identifier.")

    async def prepare(self, request: GatewayRequest, caller: str, *, workflow: bool) -> str:
        if self.mode not in (("workflow", "both") if workflow else ("conversational", "both")):
            raise GatewayError(404, "This AG-UI application mode is not published.")
        if request.resume and not workflow:
            raise GatewayError(422, "Human decisions require the workflow endpoint.")
        instance_id = scoped_id(caller, request.thread_id, "" if workflow else request.run_id)
        state = await asyncio.to_thread(self.client.get_orchestration_state, instance_id)
        payload: dict[str, str] | None = None
        if request.messages:
            text = request.messages[0].content
            payload = (
                ResearchReviewStart(topic=text).model_dump(mode="json") if workflow else
                {"session_key": scoped_id(caller, request.thread_id), "text": text}
            )
        input_hash = fingerprint(wrap_workflow_input(payload) if workflow else payload) if payload else None
        if state is not None:
            self._verify(state, workflow=workflow, input_hash=input_hash)
        elif payload is None:
            raise GatewayError(404, "AG-UI execution not found; start it with a user message.")
        else:
            try:
                if workflow:
                    await asyncio.to_thread(
                        self.workflows.start_workflow, input=payload, instance_id=instance_id,
                    )
                else:
                    await asyncio.to_thread(
                        self.client.schedule_new_orchestration, TURN_ORCHESTRATOR,
                        input=payload, instance_id=instance_id,
                    )
            except grpc.RpcError as error:
                if error.code() != grpc.StatusCode.ALREADY_EXISTS:
                    raise
                state = await asyncio.to_thread(self.client.get_orchestration_state, instance_id)
                if state is None:
                    raise GatewayError(503, "Start was already accepted; reconnect with the same identifiers.") from None
                self._verify(state, workflow=workflow, input_hash=input_hash)
        if request.resume:
            pending = await asyncio.to_thread(self.workflows.get_pending_hitl_requests, instance_id)
            decision = request.resume[0]
            if not any(item["request_id"] == decision.interrupt_id for item in pending):
                raise GatewayError(409, "The human-input request is not currently pending.")
            await asyncio.to_thread(
                self.workflows.send_hitl_response, instance_id, decision.interrupt_id,
                decision.payload.model_dump(mode="json"),
            )
        return instance_id

    async def _output(self, state: Any, instance_id: str, *, workflow: bool) -> tuple[str, list[BaseEvent]]:
        extra: list[BaseEvent] = []
        if workflow:
            output = await asyncio.to_thread(
                self.workflows.await_workflow_output, instance_id, timeout_seconds=5,
            )
            if isinstance(output, list) and len(output) == 1:
                output = output[0]
            result = ResearchReviewResult.model_validate(output)
            extra.append(CustomEvent(name="workflow_output", value=result.model_dump(mode="json")))
            return f"{result.status.capitalize()}\n\n{result.draft}", extra
        try:
            response = AgentResponse.from_dict(json.loads(state.serialized_output))
            text, presentation = response_projection(response.messages)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise GatewayError(502, "Durable assistant output does not match the response contract.") from None
        if not text and presentation is None:
            raise GatewayError(502, "Durable execution has no verifiable assistant output.")
        if presentation is not None:
            extra.append(CustomEvent(name="turn_presentation", value=presentation))
        for message in response.messages:
            if message.role != "assistant":
                continue
            for content in message.contents:
                annotations = content.to_dict().get("annotations")
                if annotations:
                    extra.append(CustomEvent(name="annotations", value={
                        "annotations": [
                            {key: value for key, value in annotation.items()
                             if key in {
                                 "type", "url", "uri", "title", "start_index", "end_index",
                                 "index", "file_id", "quote",
                             } and isinstance(value, (str, int)) and not isinstance(value, bool)}
                            for annotation in annotations
                        ],
                    }))
        return text, extra

    async def events(
        self, request: GatewayRequest, instance_id: str, *, workflow: bool,
    ) -> AsyncIterator[BaseEvent | None]:
        yield RunStartedEvent(thread_id=request.thread_id, run_id=request.run_id)
        deadline = time.monotonic() + self.timeout
        previous_status = ""
        resumed_id = request.resume[0].interrupt_id if request.resume else None
        while time.monotonic() < deadline:
            state = await asyncio.to_thread(self.client.get_orchestration_state, instance_id)
            if state is not None:
                self._verify(state, workflow=workflow)
                status = state.runtime_status.name
                if status in {"FAILED", "TERMINATED", "CANCELED", "CANCELLED"}:
                    yield RunErrorEvent(message="Durable execution did not complete successfully.", code=status)
                    return
                if status == "COMPLETED":
                    text, extra = await self._output(state, instance_id, workflow=workflow)
                    message_id = "agui-" + request.run_id
                    if text:
                        yield TextMessageStartEvent(message_id=message_id, role="assistant")
                        yield TextMessageContentEvent(message_id=message_id, delta=text)
                        yield TextMessageEndEvent(message_id=message_id)
                    for event in extra:
                        if isinstance(event, CustomEvent) and event.name == "annotations":
                            event.value["messageId"] = message_id
                        yield event
                    yield RunFinishedEvent(thread_id=request.thread_id, run_id=request.run_id)
                    return
                if workflow:
                    pending = await asyncio.to_thread(self.workflows.get_pending_hitl_requests, instance_id)
                    # The external event is asynchronous. Do not immediately re-emit
                    # the same interrupt while its accepted response is being consumed.
                    if pending and not any(item["request_id"] == resumed_id for item in pending):
                        interrupts = []
                        for item in pending:
                            yield CustomEvent(name="request_info", value={
                                "request_id": item["request_id"], "data": item["data"],
                            })
                            interrupts.append({
                                "id": item["request_id"], "reason": "input_required",
                                "message": "Review the research and approve or reject it.",
                                "responseSchema": ReviewDecision.model_json_schema(),
                            })
                        yield RunFinishedEvent(
                            thread_id=request.thread_id, run_id=request.run_id,
                            outcome=RunFinishedInterruptOutcome.model_validate({
                                "type": "interrupt", "interrupts": interrupts,
                            }),
                        )
                        return
                if status != previous_status:
                    yield CustomEvent(name="status", value={"state": status})
                    previous_status = status
                else:
                    yield None
            else:
                yield None
            await asyncio.sleep(self.poll_interval)
        yield RunErrorEvent(
            message="Polling timed out. Execution may continue in DTS; reconnect with the same thread/run identifiers.",
            code="POLL_TIMEOUT",
        )
