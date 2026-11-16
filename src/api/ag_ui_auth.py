"""Admission for trusted AG-UI frontend workloads, not end-user impersonation."""

import asyncio
import logging
from uuid import UUID

from fastapi import HTTPException
import jwt

LOGGER = logging.getLogger(__name__)


class CallerAuthorizer:
    def __init__(self, tenant_id: str, audience: str, caller_ids: str):
        self.tenant_id = str(UUID(tenant_id))
        if not audience or audience != audience.strip() or any(char.isspace() for char in audience):
            raise ValueError("AG-UI requires an explicit AG_UI_AUTH_AUDIENCE.")
        self.audience = audience
        self.callers = frozenset(str(UUID(value.strip())) for value in caller_ids.split(",") if value.strip())
        if not self.callers:
            raise ValueError("AG-UI requires AG_UI_ALLOWED_CALLER_IDS.")
        self.issuers = [
            f"https://login.microsoftonline.com/{self.tenant_id}/v2.0",
            f"https://sts.windows.net/{self.tenant_id}/",
        ]
        self.keys = jwt.PyJWKClient(
            f"https://login.microsoftonline.com/{self.tenant_id}/discovery/v2.0/keys",
            cache_keys=True, timeout=10,
        )

    def _verify(self, token: str) -> str:
        key = self.keys.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token, key, algorithms=["RS256"], audience=self.audience, issuer=self.issuers,
            options={"require": ["exp", "iat", "iss", "aud", "tid", "oid"]},
        )
        if not isinstance(claims["tid"], str) or not isinstance(claims["oid"], str):
            raise ValueError("Invalid caller identifiers.")
        tenant = str(UUID(claims["tid"]))
        caller = str(UUID(claims["oid"]))
        if tenant != self.tenant_id or caller not in self.callers:
            raise HTTPException(status_code=403, detail="AG-UI caller is not allowed.")
        return f"{tenant}:{caller}"

    async def __call__(self, authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer ") or len(authorization) > 16384:
            raise HTTPException(
                status_code=401, detail="An AG-UI caller access token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return await asyncio.to_thread(self._verify, authorization[7:])
        except jwt.PyJWKClientConnectionError:
            LOGGER.warning("AG-UI signing-key service is unavailable.")
            raise HTTPException(status_code=503, detail="Caller authentication is temporarily unavailable.") from None
        except (jwt.PyJWTError, ValueError, TypeError):
            LOGGER.warning("AG-UI caller token was rejected.")
            raise HTTPException(
                status_code=401, detail="Invalid AG-UI caller access token.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
