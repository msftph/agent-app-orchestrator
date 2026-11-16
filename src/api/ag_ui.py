"""Thin, bounded AG-UI HTTP boundary for the durable Container Apps runtime."""

from collections.abc import AsyncIterator, Awaitable, Callable
import logging
from typing import Annotated

from ag_ui.core import RunErrorEvent
from ag_ui.encoder import EventEncoder
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
import grpc
from pydantic import ValidationError

from native.ag_ui_durable import DurableGateway, GatewayError, GatewayRequest

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 65536


def create_app(
    gateway: DurableGateway, authorize: Callable[[str | None], Awaitable[str]],
) -> FastAPI:
    app = FastAPI(title="Agent Landing Zone durable AG-UI")

    async def caller(authorization: Annotated[str | None, Header()] = None) -> str:
        return await authorize(authorization)

    async def run(request: Request, identity: str, *, workflow: bool) -> StreamingResponse:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="AG-UI request is too large.")
            body.extend(chunk)
        try:
            payload = GatewayRequest.model_validate_json(bytes(body))
        except ValidationError:
            raise HTTPException(
                status_code=422,
                detail="Send one user text delta, a typed resume decision, or an empty reconnect request. Client tools/state/history are unsupported.",
            ) from None
        try:
            instance_id = await gateway.prepare(payload, identity, workflow=workflow)
        except GatewayError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from None
        except (grpc.RpcError, TimeoutError):
            LOGGER.error("AG-UI durable admission failed; execution acceptance is unconfirmed.")
            raise HTTPException(
                status_code=503,
                detail="Durable execution is unavailable or acceptance is unconfirmed. Retry with the same identifiers.",
            ) from None

        async def stream() -> AsyncIterator[str]:
            encoder = EventEncoder(accept="text/event-stream")
            try:
                async for event in gateway.events(payload, instance_id, workflow=workflow):
                    if await request.is_disconnected():
                        return
                    yield encoder.encode(event) if event is not None else ": keepalive\n\n"
            except (GatewayError, grpc.RpcError, TimeoutError, ValueError):
                LOGGER.error("AG-UI durable response could not be verified.")
                yield encoder.encode(RunErrorEvent(
                    message="Durable response could not be verified. Reconnect with the same identifiers.",
                    code="DURABLE_RESPONSE_UNAVAILABLE",
                ))

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ag-ui")
    async def converse(
        request: Request, identity: Annotated[str, Depends(caller)],
    ) -> StreamingResponse:
        return await run(request, identity, workflow=False)

    @app.post("/ag-ui/workflow")
    async def workflow(
        request: Request, identity: Annotated[str, Depends(caller)],
    ) -> StreamingResponse:
        return await run(request, identity, workflow=True)

    return app
