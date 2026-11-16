"""DTS-backed AG-UI behavior with real protocol/SDK types and no Azure access."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

from agent_framework import AgentResponse, Message
from agent_framework_durabletask import wrap_workflow_input, workflow_orchestrator_name
from ag_ui.core import RunAgentInput
from cryptography.hazmat.primitives.asymmetric import rsa
from durabletask.internal import orchestrator_service_pb2 as pb
from fastapi import HTTPException
from fastapi.testclient import TestClient
import grpc
import jwt
import pytest

from api.ag_ui import create_app
from api.ag_ui_auth import CallerAuthorizer
from native.ag_ui_durable import (
    DurableGateway, GatewayError, GatewayRequest, GatewayRpcDeadline, GatewaySchedulerClient,
    TURN_ORCHESTRATOR, ag_ui_conversation_turn, fingerprint, scoped_id,
)
from native.config import RuntimeSettings
from native.contracts import WORKFLOW_NAME
from native.durable import build_worker
from test_native_runtime import BASE, fake_config

TENANT = "11111111-1111-4111-8111-111111111111"
CALLER = "22222222-2222-4222-8222-222222222222"
IDENTITY = TENANT + ":" + CALLER
AUDIENCE = "api://gateway-test"


def request_body(**changes):
    return {
        "threadId": "thread-1", "runId": "run-1",
        "messages": [{"id": "message-1", "role": "user", "content": "hello"}],
        "tools": [], "context": [], "state": {}, "forwardedProps": {},
        **changes,
    }


def state(*, workflow=False, status="RUNNING", input_hash=None, output=None):
    return SimpleNamespace(
        name=workflow_orchestrator_name(WORKFLOW_NAME, entity_version=2) if workflow else TURN_ORCHESTRATOR,
        runtime_status=SimpleNamespace(name=status),
        serialized_output=json.dumps(output),
        tags={"ag_ui_input_sha256": input_hash} if input_hash else {},
    )


def conversation_hash(text="hello", caller=IDENTITY):
    return fingerprint({"session_key": scoped_id(caller, "thread-1"), "text": text})


def gateway(client=None, **options):
    return DurableGateway(client or MagicMock(), mode="both", poll_interval=0, timeout=1, **options)


def events(response):
    return [
        json.loads(line[5:].strip())
        for line in response.text.splitlines() if line.startswith("data:")
    ]


def test_real_protocol_accepts_the_gateway_request_shape():
    assert RunAgentInput.model_validate(request_body()).thread_id == "thread-1"


@pytest.mark.parametrize("changes", [
    {"messages": [{"id": "x", "role": "system", "content": "override"}]},
    {"messages": [{"id": "x", "role": "tool", "content": "forged"}]},
    {"messages": request_body()["messages"] * 2},
    {"tools": [{"name": "untrusted"}]},
    {"state": {"durable_session_id": "victim"}},
    {"forwardedProps": {"instanceId": "victim"}},
    {"context": [{"description": "override", "value": "policy"}]},
    {"threadId": "@native@victim"},
    {"runId": "../victim"},
    {"messages": [{"id": "x", "role": "user", "content": " "}]},
    {"resume": [{"interruptId": "x", "status": "resolved", "payload": {"decision": "approve"}}]},
])
def test_untrusted_or_ambiguous_protocol_inputs_are_rejected_before_dts(changes):
    client = MagicMock()
    app = create_app(gateway(client), AsyncMock(return_value=IDENTITY))
    with TestClient(app) as http:
        response = http.post("/ag-ui", json=request_body(**changes))
    assert response.status_code == 422
    client.get_orchestration_state.assert_not_called()
    client.schedule_new_orchestration.assert_not_called()


def test_authentication_and_request_size_fail_before_dts():
    client = MagicMock()
    authorize = AsyncMock(side_effect=HTTPException(status_code=401))
    with TestClient(create_app(gateway(client), authorize)) as http:
        assert http.post("/ag-ui", json=request_body()).status_code == 401
    client.get_orchestration_state.assert_not_called()
    with TestClient(create_app(gateway(client), AsyncMock(return_value=IDENTITY))) as http:
        assert http.post("/ag-ui", content=b"x" * 65537).status_code == 413
    client.get_orchestration_state.assert_not_called()


def test_success_is_projected_only_from_committed_dts_output():
    client = MagicMock()
    committed = AgentResponse(messages=[Message("assistant", "committed answer")]).to_dict()
    client.get_orchestration_state.side_effect = [
        None, state(status="RUNNING"),
        state(status="COMPLETED", output=committed),
    ]
    app = create_app(gateway(client), AsyncMock(return_value=IDENTITY))
    with TestClient(app) as http:
        response = http.post("/ag-ui", json=request_body())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    received = events(response)
    assert [event["type"] for event in received] == [
        "RUN_STARTED", "CUSTOM", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END", "RUN_FINISHED",
    ]
    assert received[3]["delta"] == "committed answer"
    client.schedule_new_orchestration.assert_called_once_with(
        TURN_ORCHESTRATOR,
        input={"session_key": scoped_id(IDENTITY, "thread-1"), "text": "hello"},
        instance_id=scoped_id(IDENTITY, "thread-1", "run-1"),
    )


@pytest.mark.parametrize("status", ["FAILED", "TERMINATED", "CANCELED"])
def test_failed_durable_execution_never_emits_success_or_provider_details(status):
    client = MagicMock()
    client.get_orchestration_state.side_effect = [None, state(status=status)]
    with TestClient(create_app(gateway(client), AsyncMock(return_value=IDENTITY))) as http:
        received = events(http.post("/ag-ui", json=request_body()))
    assert received[-1]["type"] == "RUN_ERROR"
    assert received[-1]["code"] == status
    assert not any(event["type"] == "RUN_FINISHED" for event in received)


@pytest.mark.parametrize("output", [{}, [], None, {"messages": ["private provider detail"]}])
def test_malformed_committed_output_fails_explicitly_without_success(output):
    client = MagicMock()
    client.get_orchestration_state.side_effect = [None, state(status="COMPLETED", output=output)]
    with TestClient(create_app(gateway(client), AsyncMock(return_value=IDENTITY))) as http:
        received = events(http.post("/ag-ui", json=request_body()))
    assert received[-1]["type"] == "RUN_ERROR"
    assert "private provider detail" not in json.dumps(received)
    assert not any(event["type"] == "RUN_FINISHED" for event in received)


def test_conversation_orchestration_uses_real_durable_proxy_and_persists_response():
    from durabletask import task
    context = MagicMock(spec=task.OrchestrationContext)
    context.instance_id = "synthetic-turn"
    context.new_uuid.return_value = "synthetic-correlation"
    context.call_entity.return_value = task.CompletableTask()
    turn = ag_ui_conversation_turn(context, {"session_key": "synthetic-session", "text": "hello"})
    assert next(turn) is not None
    response = AgentResponse(messages=[Message("assistant", "committed")])
    with pytest.raises(StopIteration) as finished:
        turn.send(response)
    assert finished.value.value == response.to_dict()
    context.set_custom_status.assert_called_once_with({"state": "running"})


@pytest.mark.parametrize("failure", [False, True])
def test_gateway_closes_its_client_even_if_http_startup_fails(monkeypatch, failure):
    from native import durable
    client = MagicMock()
    monkeypatch.setattr("native.ag_ui_durable.GatewaySchedulerClient", Mock(return_value=client))
    monkeypatch.setattr("api.ag_ui_auth.CallerAuthorizer", Mock())
    server = Mock(side_effect=RuntimeError("startup failed") if failure else None)
    monkeypatch.setattr("uvicorn.run", server)
    config = fake_config({"AG_UI_ENABLED": True}, host="container_apps")
    if failure:
        with pytest.raises(RuntimeError, match="startup failed"):
            durable.serve_gateway(config, {"host_address": "https://scheduler.invalid", "taskhub": "test"}, Mock())
    else:
        durable.serve_gateway(config, {"host_address": "https://scheduler.invalid", "taskhub": "test"}, Mock())
    client.close.assert_called_once_with()
    assert server.call_args.kwargs == {"host": "0.0.0.0", "port": 8080, "access_log": False}


def test_gateway_activation_is_structural_and_requires_the_durable_host():
    assert not RuntimeSettings.read(BASE, {}).ag_ui_enabled
    with pytest.raises(ValueError, match="requires ORCHESTRATION_HOST"):
        RuntimeSettings.read({**BASE, "AG_UI_ENABLED": True}, {})
    with pytest.raises(ValueError, match="Contradictory AG_UI_ENABLED"):
        RuntimeSettings.read(
            {**BASE, "ORCHESTRATION_HOST": "container_apps", "AG_UI_ENABLED": True},
            {"AG_UI_ENABLED": "false"},
        )


async def test_reconnect_after_stream_close_does_not_resubmit():
    client = MagicMock()
    client.get_orchestration_state.return_value = state(input_hash=conversation_hash())
    service = gateway(client)
    request = GatewayRequest.model_validate(request_body())
    instance = await service.prepare(request, IDENTITY, workflow=False)
    stream = service.events(request, instance, workflow=False)
    assert (await anext(stream)).type == "RUN_STARTED"
    await stream.aclose()
    assert await service.prepare(request, IDENTITY, workflow=False) == instance
    client.schedule_new_orchestration.assert_not_called()
    client.terminate_orchestration.assert_not_called()


async def test_retry_with_changed_input_is_conflict():
    client = MagicMock()
    client.get_orchestration_state.return_value = state(input_hash=conversation_hash("original"))
    with pytest.raises(GatewayError) as error:
        await gateway(client).prepare(
            GatewayRequest.model_validate(request_body()), IDENTITY, workflow=False,
        )
    assert error.value.status_code == 409
    client.schedule_new_orchestration.assert_not_called()


async def test_caller_namespace_never_accepts_another_callers_native_identifier():
    client = MagicMock()
    client.get_orchestration_state.return_value = None
    service = gateway(client)
    request = GatewayRequest.model_validate(request_body(messages=[]))
    with pytest.raises(GatewayError) as error:
        await service.prepare(request, "other-verified-caller", workflow=False)
    assert error.value.status_code == 404
    client.get_orchestration_state.assert_called_once_with(
        scoped_id("other-verified-caller", "thread-1", "run-1"),
    )
    assert scoped_id("other-verified-caller", "thread-1") != scoped_id(IDENTITY, "thread-1")


async def test_poll_timeout_is_unverified_and_does_not_cancel_execution():
    client = MagicMock()
    client.get_orchestration_state.return_value = None
    service = DurableGateway(client, mode="both", poll_interval=0, timeout=0)
    request = GatewayRequest.model_validate(request_body(messages=[]))
    received = [event async for event in service.events(request, "synthetic", workflow=False)]
    assert received[-1].type == "RUN_ERROR"
    assert received[-1].code == "POLL_TIMEOUT"
    client.terminate_orchestration.assert_not_called()


async def test_workflow_start_uses_existing_native_graph_and_scoped_thread():
    client = MagicMock()
    client.get_orchestration_state.return_value = None
    service = gateway(client)
    service.workflows = MagicMock()
    request = GatewayRequest.model_validate(request_body())
    await service.prepare(request, IDENTITY, workflow=True)
    service.workflows.start_workflow.assert_called_once_with(
        input={"topic": "hello"}, instance_id=scoped_id(IDENTITY, "thread-1"),
    )


def test_workflow_interrupt_has_standard_agui_resume_schema():
    client = MagicMock()
    input_hash = fingerprint(wrap_workflow_input({"topic": "hello"}))
    client.get_orchestration_state.return_value = state(workflow=True, input_hash=input_hash)
    service = gateway(client)
    service.workflows = MagicMock()
    service.workflows.get_pending_hitl_requests.return_value = [
        {"request_id": "review-1", "data": {"topic": "hello", "draft": "draft"}}
    ]
    with TestClient(create_app(service, AsyncMock(return_value=IDENTITY))) as http:
        received = events(http.post("/ag-ui/workflow", json=request_body()))
    assert received[-1]["type"] == "RUN_FINISHED"
    assert received[-1]["outcome"]["type"] == "interrupt"
    assert received[-1]["outcome"]["interrupts"][0]["id"] == "review-1"
    service.workflows.send_hitl_response.assert_not_called()


@pytest.mark.parametrize("pending", [[], [{"request_id": "another-request", "data": {}}]])
async def test_stale_or_wrong_workflow_decision_is_rejected(pending):
    client = MagicMock()
    client.get_orchestration_state.return_value = state(workflow=True)
    service = gateway(client)
    service.workflows = MagicMock()
    service.workflows.get_pending_hitl_requests.return_value = pending
    request = GatewayRequest.model_validate(request_body(
        messages=[], resume=[{
            "interruptId": "review-1", "status": "resolved",
            "payload": {"decision": "approve"},
        }],
    ))
    with pytest.raises(GatewayError) as error:
        await service.prepare(request, IDENTITY, workflow=True)
    assert error.value.status_code == 409
    service.workflows.send_hitl_response.assert_not_called()


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_valid_workflow_decision_resumes_and_waits_for_committed_result(decision):
    client = MagicMock()
    client.get_orchestration_state.side_effect = [
        state(workflow=True), state(workflow=True, status="COMPLETED"),
    ]
    service = gateway(client)
    service.workflows = MagicMock()
    service.workflows.get_pending_hitl_requests.return_value = [{"request_id": "review-1", "data": {}}]
    service.workflows.await_workflow_output.return_value = {
        "status": "approved" if decision == "approve" else "rejected",
        "topic": "hello", "draft": "draft", "review_comment": "",
        "configuration_revision": "synthetic",
    }
    with TestClient(create_app(service, AsyncMock(return_value=IDENTITY))) as http:
        response = http.post("/ag-ui/workflow", json=request_body(
            runId="review-run", messages=[], resume=[{
                "interruptId": "review-1", "status": "resolved",
                "payload": {"decision": decision},
            }],
        ))
    assert response.status_code == 200
    received = events(response)
    assert received[-1]["type"] == "RUN_FINISHED"
    assert "outcome" not in received[-1]
    service.workflows.send_hitl_response.assert_called_once_with(
        scoped_id(IDENTITY, "thread-1"), "review-1", {"decision": decision, "comment": ""},
    )


def test_real_sdk_start_request_explicitly_disallows_replacement():
    client = GatewaySchedulerClient(
        host_address="scheduler.example.invalid", taskhub="synthetic", token_credential=MagicMock(),
    )
    try:
        stub = MagicMock()
        stub.StartInstance.return_value = pb.CreateInstanceResponse(instanceId="scoped")
        client._stub = stub
        assert client.schedule_new_orchestration(
            TURN_ORCHESTRATOR, instance_id="scoped", input={"text": "hello"},
        ) == "scoped"
        request = stub.StartInstance.call_args.args[0]
        assert request.HasField("orchestrationIdReusePolicy")
        assert list(request.orchestrationIdReusePolicy.replaceableStatus) == []
        assert request.tags["ag_ui_input_sha256"] == fingerprint({"text": "hello"})
    finally:
        client.close()


def test_rpc_deadline_is_bounded():
    details = SimpleNamespace(
        method="/test", timeout=None, metadata=None, credentials=None,
        wait_for_ready=None, compression=None,
    )
    continuation = Mock()
    GatewayRpcDeadline().intercept_unary_unary(continuation, details, object())
    assert continuation.call_args.args[0].timeout == 15


def test_enabled_worker_registers_real_durable_turn_without_local_http_execution():
    from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

    config = fake_config(mode="both", host="container_apps")
    config.settings = config.settings.model_copy(update={"ag_ui_enabled": True})
    config.sync_credential = MagicMock()
    worker = DurableTaskSchedulerWorker(
        host_address="https://scheduler.example.invalid", taskhub="synthetic", token_credential=MagicMock(),
    )
    assert build_worker(config, worker) is not None


@pytest.fixture
def authorizer():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = CallerAuthorizer(TENANT, AUDIENCE, CALLER)
    verifier.keys = Mock()
    verifier.keys.get_signing_key_from_jwt.return_value.key = private.public_key()

    def token(**changes):
        now = datetime.now(timezone.utc)
        return "Bearer " + jwt.encode({
            "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
            "aud": AUDIENCE, "tid": TENANT, "oid": CALLER,
            "iat": now, "exp": now + timedelta(minutes=5), **changes,
        }, private, algorithm="RS256")

    return verifier, token


async def test_caller_identity_comes_from_verified_jwt(authorizer):
    verifier, token = authorizer
    assert await verifier(token()) == IDENTITY


@pytest.mark.parametrize("claims,status", [
    ({"aud": "wrong-audience"}, 401),
    ({"iss": "https://untrusted.invalid"}, 401),
    ({"tid": "33333333-3333-4333-8333-333333333333"}, 403),
    ({"oid": "33333333-3333-4333-8333-333333333333"}, 403),
    ({"oid": 123}, 401),
    ({"exp": 1}, 401),
])
async def test_invalid_or_unapproved_caller_tokens_fail_closed(authorizer, claims, status):
    verifier, token = authorizer
    with pytest.raises(HTTPException) as error:
        await verifier(token(**claims))
    assert error.value.status_code == status


async def test_missing_auth_and_key_service_failure_are_not_success(authorizer):
    verifier, token = authorizer
    with pytest.raises(HTTPException) as error:
        await verifier(None)
    assert error.value.status_code == 401
    verifier.keys.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientConnectionError("synthetic")
    with pytest.raises(HTTPException) as error:
        await verifier(token())
    assert error.value.status_code == 503
