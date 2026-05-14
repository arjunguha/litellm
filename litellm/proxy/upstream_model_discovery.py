"""
Dynamic model discovery for upstream OpenAI-compatible servers.

Configured via the `upstream_model_discovery` YAML section. Each entry points
to an upstream OpenAI-compatible server; on startup (and on a refresh
interval), the proxy queries its `/v1/models` endpoint and registers every
model it finds as `{prefix}/{model_id}` in the local router.

Example yaml:

    upstream_model_discovery:
      - api_base: https://upstream-litellm.example.com/v1
        api_key: os.environ/UPSTREAM_LITELLM_KEY
        # optional: override the auto-derived prefix
        prefix: upstream-litellm
        # optional: refresh cadence in seconds; 0 disables refresh
        refresh_interval: 300

The operator never whitelists individual upstream models — any model the
upstream serves at refresh time becomes available locally.
"""

from __future__ import annotations

import asyncio
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import get_secret

_DEFAULT_REFRESH_INTERVAL = 300


def normalize_api_base_to_prefix(api_base: str) -> str:
    """Derive a deterministic model-name prefix from an api_base URL.

    Strips scheme/path and userinfo, keeps host *and* port, then replaces
    non-alphanumerics with hyphens — so two upstreams sharing a hostname
    on different ports (e.g. `localhost:4000` vs `localhost:5000`) get
    distinct prefixes. Examples:

    - `https://upstream-litellm.example.com/v1`     -> `upstream-litellm-example-com`
    - `https://upstream-litellm.example.com:4000/v1`-> `upstream-litellm-example-com-4000`
    - `http://localhost:5000/v1`                    -> `localhost-5000`
    """
    parsed = urlparse(api_base)
    if parsed.hostname:
        host_port = parsed.hostname
        if parsed.port is not None:
            host_port = f"{host_port}:{parsed.port}"
    else:
        # No scheme parsed — fall back to netloc, then the raw input. Strip any
        # userinfo (`user:pass@`) so secrets never leak into the prefix.
        host_port = (parsed.netloc or api_base).rsplit("@", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", host_port).strip("-").lower()
    return slug or "upstream"


def _resolve_secret(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret(value)
    return value


class UpstreamModelDiscovery:
    """Discovers and re-discovers models from one or more upstream servers.

    Holds the static (yaml-declared) model_list so each refresh recomputes
    the union [static + discovered]. Static entries win on name collisions.
    """

    def __init__(
        self,
        upstreams: List[Dict[str, Any]],
        static_model_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not isinstance(upstreams, list) or len(upstreams) == 0:
            raise ValueError(
                "upstream_model_discovery must be a non-empty list of upstream entries"
            )

        self.upstreams: List[Dict[str, Any]] = []
        for entry in upstreams:
            api_base = _resolve_secret(entry.get("api_base"))
            if not api_base or not isinstance(api_base, str):
                raise ValueError(
                    "upstream_model_discovery entry is missing required string 'api_base'"
                )
            api_key = _resolve_secret(entry.get("api_key"))
            prefix = entry.get("prefix") or normalize_api_base_to_prefix(api_base)
            refresh_interval = int(
                entry.get("refresh_interval", _DEFAULT_REFRESH_INTERVAL)
            )
            self.upstreams.append(
                {
                    "api_base": api_base.rstrip("/"),
                    "api_key": api_key,
                    "prefix": prefix,
                    "refresh_interval": refresh_interval,
                }
            )

        self.static_model_list: List[Dict[str, Any]] = deepcopy(static_model_list or [])
        self._refresh_task: Optional[asyncio.Task] = None

    async def _fetch_upstream_models(
        self, api_base: str, api_key: Optional[str]
    ) -> List[str]:
        url = f"{api_base.rstrip('/')}/models"
        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        return [m["id"] for m in models if isinstance(m, dict) and "id" in m]

    def _build_entries_for_upstream(
        self, upstream: Dict[str, Any], model_ids: List[str]
    ) -> List[Dict[str, Any]]:
        prefix = upstream["prefix"]
        api_base = upstream["api_base"]
        api_key = upstream["api_key"]
        entries: List[Dict[str, Any]] = []
        for model_id in model_ids:
            entries.append(
                {
                    "model_name": f"{prefix}/{model_id}",
                    "litellm_params": {
                        "model": f"openai/{model_id}",
                        "api_base": api_base,
                        "api_key": api_key,
                    },
                    "model_info": {
                        "upstream_discovered": True,
                        "upstream_api_base": api_base,
                        "upstream_prefix": prefix,
                    },
                }
            )
        return entries

    async def discover(self) -> List[Dict[str, Any]]:
        """Query every configured upstream and return new model_list entries.

        A failure against any single upstream is logged and that upstream's
        entries are simply omitted from this round — the others still load.
        """
        discovered: List[Dict[str, Any]] = []
        for upstream in self.upstreams:
            try:
                ids = await self._fetch_upstream_models(
                    upstream["api_base"], upstream["api_key"]
                )
                discovered.extend(self._build_entries_for_upstream(upstream, ids))
                verbose_proxy_logger.info(
                    "upstream_model_discovery: discovered %d models from %s (prefix=%s)",
                    len(ids),
                    upstream["api_base"],
                    upstream["prefix"],
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    "upstream_model_discovery: failed to fetch models from %s: %s",
                    upstream["api_base"],
                    e,
                )
        return discovered

    async def build_combined_model_list(self) -> List[Dict[str, Any]]:
        """Return static yaml-declared models plus freshly-discovered ones."""
        discovered = await self.discover()
        static_names = {m.get("model_name") for m in self.static_model_list}
        deduped = [m for m in discovered if m["model_name"] not in static_names]
        return deepcopy(self.static_model_list) + deduped

    def start_refresh_task(self, router: Any) -> None:
        """Schedule the periodic refresh loop against ``router``.

        Cancels any previously-running refresh task on this manager so config
        reloads don't accumulate stale loops. Uses the smallest non-zero
        ``refresh_interval`` across upstreams; if every upstream has
        ``refresh_interval: 0``, no loop is started.
        """
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None

        intervals = [
            u["refresh_interval"] for u in self.upstreams if u["refresh_interval"] > 0
        ]
        if not intervals:
            return
        interval = min(intervals)
        self._refresh_task = asyncio.create_task(self._refresh_loop(router, interval))

    async def _refresh_loop(self, router: Any, interval: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                model_list = await self.build_combined_model_list()
                router.set_model_list(model_list)
                verbose_proxy_logger.info(
                    "upstream_model_discovery: refreshed model list (total=%d)",
                    len(model_list),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                verbose_proxy_logger.warning(
                    "upstream_model_discovery: refresh error: %s", e
                )
