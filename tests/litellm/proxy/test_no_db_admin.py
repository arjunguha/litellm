from pathlib import Path

import pytest

import litellm
from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _no_db_key_generation_helper,
)
from litellm.proxy._types import hash_token
from litellm.proxy.no_db_admin import (
    NoDBAdminStore,
    get_no_db_admin_store,
    initialize_no_db_admin_store,
)


def test_no_db_admin_store_reads_config_and_shows_configured_key(tmp_path: Path):
    secrets_file = tmp_path / "secrets.yaml"
    store = NoDBAdminStore(
        users_config=[
            {
                "user_id": "admin-user",
                "user_email": "admin@example.com",
                "user_role": "proxy_admin",
                "teams": ["team-a"],
            }
        ],
        teams_config=[
            {
                "team_id": "team-a",
                "team_alias": "Team A",
                "members_with_roles": [{"user_id": "admin-user", "role": "admin"}],
            }
        ],
        secrets_file_path=str(secrets_file),
    )

    store.set_user_password("admin-user", "fresh-password")
    store.upsert_api_key(
        key_alias="local-dev",
        key="sk-local-dev",
        user_id="admin-user",
        team_id="team-a",
    )

    user = store.get_user_by_email("admin@example.com")
    assert user is not None
    assert store.verify_user_password(user, "fresh-password") is True

    key_auth = store.get_key_auth_by_hash(hash_token("sk-local-dev"))
    assert key_auth is not None
    serialized_key = store.serialize_key_auth(key_auth)
    assert serialized_key["key"] == "sk-local-dev"
    assert serialized_key["token_id"] == hash_token("sk-local-dev")
    assert serialized_key["user_email"] == "admin@example.com"
    assert serialized_key["team_alias"] == "Team A"


def test_initialize_no_db_admin_store_uses_config_relative_secrets_path(
    tmp_path: Path,
):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("general_settings:\n  no_database_admin_ui: true\n")

    store = initialize_no_db_admin_store(
        config={
            "users": [{"user_id": "u1", "user_role": "proxy_admin"}],
            "teams": [],
        },
        config_file_path=str(config_file),
        general_settings={"no_database_admin_ui": True},
    )

    assert store is get_no_db_admin_store()
    assert store is not None
    assert store.secrets_file_path == str(tmp_path / "secrets.yaml")
    assert store.get_user("u1") is not None


@pytest.mark.asyncio
async def test_no_db_key_generation_writes_secret_with_runtime_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "key_generation_settings", None)
    monkeypatch.setattr(litellm, "default_key_generate_params", None)
    monkeypatch.setattr(litellm, "upperbound_key_generate_params", None)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("general_settings:\n  no_database_admin_ui: true\n")
    store = initialize_no_db_admin_store(
        config={
            "users": [
                {
                    "user_id": "user-1",
                    "user_email": "user@example.com",
                    "user_role": "internal_user",
                    "teams": ["team-a"],
                }
            ],
            "teams": [
                {
                    "team_id": "team-a",
                    "team_alias": "Team A",
                    "members_with_roles": [{"user_id": "user-1", "role": "user"}],
                    "team_member_permissions": ["/key/generate"],
                }
            ],
        },
        config_file_path=str(config_file),
        general_settings={"no_database_admin_ui": True},
    )
    assert store is not None

    response = await _no_db_key_generation_helper(
        data=GenerateKeyRequest(
            key_alias="ui-created",
            key="sk-ui-created",
            team_id="team-a",
            models=["gpt-4o"],
            max_budget=10,
            permissions={"allow_pii_controls": True},
            metadata={"app": "dashboard"},
        ),
        user_api_key_dict=UserAPIKeyAuth(
            token="session-token",
            user_id="user-1",
            user_role=LitellmUserRoles.INTERNAL_USER,
        ),
        litellm_changed_by=None,
    )

    assert response.key == "sk-ui-created"
    assert response.token_id == hash_token("sk-ui-created")
    key_auth = store.get_key_auth_by_hash(hash_token("sk-ui-created"))
    assert key_auth is not None
    assert key_auth.user_id == "user-1"
    assert key_auth.team_id == "team-a"
    assert key_auth.models == ["gpt-4o"]
    assert key_auth.max_budget == 10
    assert key_auth.permissions == {"allow_pii_controls": True}
    assert key_auth.metadata["app"] == "dashboard"
