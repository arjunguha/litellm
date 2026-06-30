from pathlib import Path

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
