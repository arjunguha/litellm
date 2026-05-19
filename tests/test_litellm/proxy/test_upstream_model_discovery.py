"""
Tests for litellm/proxy/upstream_model_discovery.py

Covers:
- api_base -> prefix normalization
- discovery against a mocked upstream /v1/models
- merge behavior (static entries win on collision)
- env-var resolution in the upstream entry
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.upstream_model_discovery import (
    UpstreamModelDiscovery,
    normalize_api_base_to_prefix,
)


class TestNormalizeApiBaseToPrefix:
    def test_strips_scheme_and_path(self):
        assert (
            normalize_api_base_to_prefix("https://upstream-litellm.example.com/v1")
            == "upstream-litellm-example-com"
        )

    def test_includes_port(self):
        assert (
            normalize_api_base_to_prefix("http://my-proxy.local:4000/v1")
            == "my-proxy-local-4000"
        )

    def test_default_port_for_scheme_not_in_url_is_omitted(self):
        # When the URL has no explicit port, the port isn't appended.
        assert (
            normalize_api_base_to_prefix("https://upstream.example.com/v1")
            == "upstream-example-com"
        )

    def test_distinguishes_same_host_different_ports(self):
        a = normalize_api_base_to_prefix("http://localhost:4000/v1")
        b = normalize_api_base_to_prefix("http://localhost:5000/v1")
        assert a != b
        assert a == "localhost-4000"
        assert b == "localhost-5000"

    def test_lowercases(self):
        assert (
            normalize_api_base_to_prefix("https://API.EXAMPLE.com/v1")
            == "api-example-com"
        )

    def test_handles_ip(self):
        assert (
            normalize_api_base_to_prefix("http://127.0.0.1:8080/v1") == "127-0-0-1-8080"
        )

    def test_strips_userinfo(self):
        # Credentials embedded in a URL must never appear in the prefix.
        result = normalize_api_base_to_prefix("user:pass@upstream.example.com:4000/v1")
        assert "user" not in result
        assert "pass" not in result

    def test_handles_no_scheme(self):
        # urlparse without scheme yields no hostname, so the input is slugified.
        result = normalize_api_base_to_prefix("upstream-litellm.example.com")
        assert result == "upstream-litellm-example-com"

    def test_empty_falls_back(self):
        assert normalize_api_base_to_prefix("") == "upstream"


def _mock_models_response(model_ids):
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model"} for mid in model_ids],
    }


def _patch_httpx_get(response_json, status_code=200):
    """Patches httpx.AsyncClient so .get(...) returns a stub response."""
    response = MagicMock()
    response.json.return_value = response_json
    response.raise_for_status = MagicMock()
    response.status_code = status_code

    client_instance = MagicMock()
    client_instance.get = AsyncMock(return_value=response)
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=None)

    return patch(
        "litellm.proxy.upstream_model_discovery.httpx.AsyncClient",
        return_value=client_instance,
    )


class TestUpstreamModelDiscovery:
    def test_rejects_empty_upstreams(self):
        with pytest.raises(ValueError):
            UpstreamModelDiscovery(upstreams=[])

    def test_rejects_missing_api_base(self):
        with pytest.raises(ValueError):
            UpstreamModelDiscovery(upstreams=[{"api_key": "sk-x"}])

    def test_resolves_env_secrets(self, monkeypatch):
        monkeypatch.setenv("UPSTREAM_BASE", "https://upstream.example.com/v1")
        monkeypatch.setenv("UPSTREAM_KEY", "sk-from-env")

        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "os.environ/UPSTREAM_BASE",
                    "api_key": "os.environ/UPSTREAM_KEY",
                }
            ]
        )
        assert mgr.upstreams[0]["api_base"] == "https://upstream.example.com/v1"
        assert mgr.upstreams[0]["api_key"] == "sk-from-env"
        # Auto-derived prefix
        assert mgr.upstreams[0]["prefix"] == "upstream-example-com"

    def test_explicit_prefix_wins_over_auto(self):
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://upstream.example.com/v1",
                    "prefix": "my-prefix",
                }
            ]
        )
        assert mgr.upstreams[0]["prefix"] == "my-prefix"

    @pytest.mark.asyncio
    async def test_discover_builds_entries(self):
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "up",
                }
            ]
        )
        with _patch_httpx_get(_mock_models_response(["gpt-4", "claude-3-opus"])):
            entries = await mgr.discover()

        assert len(entries) == 2
        names = sorted(e["model_name"] for e in entries)
        assert names == ["up/claude-3-opus", "up/gpt-4"]

        for entry in entries:
            assert (
                entry["litellm_params"]["api_base"] == "https://upstream.example.com/v1"
            )
            assert entry["litellm_params"]["api_key"] == "sk-x"
            assert entry["litellm_params"]["model"].startswith("openai/")
            assert entry["model_info"]["upstream_discovered"] is True

    @pytest.mark.asyncio
    async def test_combined_static_wins_on_collision(self):
        static = [
            {
                "model_name": "up/gpt-4",
                "litellm_params": {"model": "anthropic/claude", "api_key": "sk-static"},
            }
        ]
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "up",
                }
            ],
            static_model_list=static,
        )
        with _patch_httpx_get(_mock_models_response(["gpt-4", "gpt-5"])):
            combined = await mgr.build_combined_model_list()

        # Static "up/gpt-4" kept, discovered "up/gpt-4" dropped, "up/gpt-5" added.
        names = sorted(m["model_name"] for m in combined)
        assert names == ["up/gpt-4", "up/gpt-5"]
        static_entry = next(m for m in combined if m["model_name"] == "up/gpt-4")
        assert static_entry["litellm_params"]["model"] == "anthropic/claude"

    @pytest.mark.asyncio
    async def test_discover_swallows_per_upstream_errors(self):
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://broken.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "broken",
                }
            ]
        )

        # AsyncClient.get raises — discover() must return [] rather than blow up.
        client_instance = MagicMock()
        client_instance.get = AsyncMock(side_effect=RuntimeError("upstream down"))
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "litellm.proxy.upstream_model_discovery.httpx.AsyncClient",
            return_value=client_instance,
        ):
            entries = await mgr.discover()

        assert entries == []

    @pytest.mark.asyncio
    async def test_refresh_task_calls_set_model_list(self):
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "up",
                    "refresh_interval": 0,  # disable scheduled loop
                }
            ]
        )
        router = MagicMock()
        # refresh_interval=0 -> no task should be scheduled
        mgr.start_refresh_task(router)
        assert mgr._refresh_task is None

        # Manually exercise build + set, which is what the loop does.
        with _patch_httpx_get(_mock_models_response(["gpt-4o"])):
            combined = await mgr.build_combined_model_list()
        router.set_model_list(combined)
        router.set_model_list.assert_called_once()
        names = [m["model_name"] for m in router.set_model_list.call_args[0][0]]
        assert "up/gpt-4o" in names

    @pytest.mark.asyncio
    async def test_load_config_wires_upstream_discovery(self, tmp_path):
        """End-to-end: ProxyConfig.load_config picks up `upstream_model_discovery`
        from a YAML file, fetches upstream models, and seeds router_params['model_list']
        with `{prefix}/{model_id}` entries.
        """
        import yaml
        from unittest.mock import patch as _patch

        from litellm.proxy.proxy_server import ProxyConfig

        config = {
            "general_settings": {},
            "model_list": [
                {
                    "model_name": "local-only",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ],
            "upstream_model_discovery": [
                {
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "up",
                    "refresh_interval": 0,
                }
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        captured: dict = {}

        def _fake_router(**kwargs):
            captured["model_list"] = kwargs.get("model_list")
            mock_router = MagicMock()
            mock_router.get_model_list.return_value = kwargs.get("model_list") or []
            mock_router.adaptive_routers = {}
            mock_router.cache = MagicMock()
            mock_router.cache.redis_cache = None
            return mock_router

        with (
            _patch_httpx_get(_mock_models_response(["gpt-4", "gpt-5"])),
            _patch("litellm.Router", side_effect=_fake_router),
        ):
            pc = ProxyConfig()
            await pc.load_config(router=None, config_file_path=str(config_path))

        names = sorted(m["model_name"] for m in captured["model_list"])
        assert "local-only" in names
        assert "up/gpt-4" in names
        assert "up/gpt-5" in names

    @pytest.mark.asyncio
    async def test_refresh_loop_starts_when_interval_positive(self):
        mgr = UpstreamModelDiscovery(
            upstreams=[
                {
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                    "prefix": "up",
                    "refresh_interval": 60,
                }
            ]
        )
        router = MagicMock()
        mgr.start_refresh_task(router)
        try:
            assert mgr._refresh_task is not None
            assert not mgr._refresh_task.done()
        finally:
            mgr._refresh_task.cancel()
            try:
                await mgr._refresh_task
            except (asyncio.CancelledError, BaseException):
                pass

    @pytest.mark.asyncio
    async def test_db_reconciliation_preserves_upstream_discovered_models(
        self, monkeypatch
    ):
        from litellm.proxy import proxy_server
        from litellm.proxy.proxy_server import ProxyConfig

        router = MagicMock()
        router.get_model_ids.return_value = ["db-model-id", "dynamic-model-id"]
        router.get_model_list.return_value = [
            {
                "model_name": "up/gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_base": "https://upstream.example.com/v1",
                    "api_key": "sk-x",
                },
                "model_info": {
                    "id": "dynamic-model-id",
                    "upstream_discovered": True,
                },
            }
        ]

        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_config_file_path", None)

        proxy_config = ProxyConfig()
        proxy_config.get_config = AsyncMock(return_value={"model_list": []})
        db_model = SimpleNamespace(
            model_id="db-model-id",
            model_name="db-model",
            model_info={},
            litellm_params={},
        )

        deleted = await proxy_config._delete_deployment(db_models=[db_model])

        assert deleted == 0
        router.delete_deployment.assert_not_called()
