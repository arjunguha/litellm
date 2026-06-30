import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml

import litellm
from litellm._uuid import uuid
from litellm.constants import LITELLM_PROXY_ADMIN_NAME, LITELLM_UI_SESSION_DURATION
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
    hash_token,
)


NO_DB_ADMIN_UI_SETTING = "no_database_admin_ui"
NO_DB_SECRETS_FILE_SETTING = "secrets_file"


_store: Optional["NoDBAdminStore"] = None


def is_no_db_admin_enabled(general_settings: Optional[dict]) -> bool:
    if general_settings is None:
        return False
    return general_settings.get(NO_DB_ADMIN_UI_SETTING) is True


def get_no_db_admin_store() -> Optional["NoDBAdminStore"]:
    return _store


def initialize_no_db_admin_store(
    *,
    config: dict,
    config_file_path: Optional[str],
    general_settings: dict,
) -> Optional["NoDBAdminStore"]:
    global _store
    if not is_no_db_admin_enabled(general_settings):
        _store = None
        return None

    configured_path = general_settings.get(NO_DB_SECRETS_FILE_SETTING)
    if configured_path is None:
        base_dir = os.path.dirname(os.path.abspath(config_file_path or "."))
        configured_path = os.path.join(base_dir, "secrets.yaml")
    elif not os.path.isabs(configured_path):
        base_dir = os.path.dirname(os.path.abspath(config_file_path or "."))
        configured_path = os.path.join(base_dir, configured_path)

    _store = NoDBAdminStore(
        users_config=config.get("users") or [],
        teams_config=config.get("teams") or [],
        secrets_file_path=configured_path,
    )
    return _store


def no_db_admin_error() -> Exception:
    from litellm.proxy._types import ProxyErrorTypes, ProxyException

    return ProxyException(
        message=(
            "This LiteLLM proxy is running with no_database_admin_ui enabled. "
            "Manage users, teams, passwords, and API keys in config.yaml and secrets.yaml."
        ),
        type=ProxyErrorTypes.bad_request_error,
        param=NO_DB_ADMIN_UI_SETTING,
        code=400,
    )


def _duration_to_timedelta(duration: Optional[str]) -> timedelta:
    if duration is None:
        return timedelta(hours=24)
    value = duration.strip()
    if not value:
        return timedelta(hours=24)
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    return timedelta(seconds=int(value))


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


class NoDBAdminStore:
    def __init__(
        self,
        *,
        users_config: List[dict],
        teams_config: List[dict],
        secrets_file_path: str,
    ) -> None:
        self.secrets_file_path = secrets_file_path
        self._users_config = users_config
        self._teams_config = teams_config
        self._secrets = self._read_secrets()
        self.users = self._build_users(users_config)
        self.teams = self._build_teams(teams_config)

    def _read_secrets(self) -> dict:
        if not os.path.exists(self.secrets_file_path):
            return {"users": {}, "keys": {}, "ui_sessions": {}}
        with open(self.secrets_file_path, "r") as secrets_file:
            data = yaml.safe_load(secrets_file) or {}
        data.setdefault("users", {})
        data.setdefault("keys", {})
        data.setdefault("ui_sessions", {})
        return data

    def _write_secrets(self) -> None:
        os.makedirs(
            os.path.dirname(os.path.abspath(self.secrets_file_path)), exist_ok=True
        )
        with open(self.secrets_file_path, "w") as secrets_file:
            yaml.safe_dump(self._secrets, secrets_file, default_flow_style=False)

    def reload(self) -> None:
        self._secrets = self._read_secrets()

    def _build_users(self, users_config: List[dict]) -> Dict[str, LiteLLM_UserTable]:
        users: Dict[str, LiteLLM_UserTable] = {}
        for user in users_config:
            user_id = user.get("user_id") or user.get("user_email")
            if not user_id:
                continue
            users[user_id] = LiteLLM_UserTable(
                user_id=user_id,
                user_email=user.get("user_email"),
                user_alias=user.get("user_alias"),
                user_role=user.get("user_role", LitellmUserRoles.INTERNAL_USER),
                teams=user.get("teams") or [],
                models=user.get("models") or [],
                max_budget=user.get("max_budget"),
                tpm_limit=user.get("tpm_limit"),
                rpm_limit=user.get("rpm_limit"),
                metadata=user.get("metadata"),
                created_at=_parse_datetime(user.get("created_at")),
                updated_at=_parse_datetime(user.get("updated_at")),
            )
        return users

    def _build_teams(self, teams_config: List[dict]) -> Dict[str, LiteLLM_TeamTable]:
        teams: Dict[str, LiteLLM_TeamTable] = {}
        for team in teams_config:
            team_id = team.get("team_id")
            if not team_id:
                continue
            members_with_roles = team.get("members_with_roles")
            if members_with_roles is None:
                members_with_roles = [
                    {"user_id": user_id, "role": "user"}
                    for user_id in team.get("members", [])
                ]
            teams[team_id] = LiteLLM_TeamTable(
                team_id=team_id,
                team_alias=team.get("team_alias"),
                organization_id=team.get("organization_id"),
                admins=team.get("admins") or [],
                members=team.get("members") or [],
                members_with_roles=members_with_roles,
                team_member_permissions=team.get("team_member_permissions"),
                metadata=team.get("metadata"),
                tpm_limit=team.get("tpm_limit"),
                rpm_limit=team.get("rpm_limit"),
                max_budget=team.get("max_budget"),
                soft_budget=team.get("soft_budget"),
                models=team.get("models") or [],
                blocked=team.get("blocked", False),
                created_at=_parse_datetime(team.get("created_at")),
                updated_at=_parse_datetime(team.get("updated_at")),
            )
        return teams

    def get_user_by_email(self, email: str) -> Optional[LiteLLM_UserTable]:
        email_lower = email.lower()
        for user in self.users.values():
            if user.user_email and user.user_email.lower() == email_lower:
                return user
        return None

    def get_user(self, user_id: Optional[str]) -> Optional[LiteLLM_UserTable]:
        if user_id is None:
            return None
        return self.users.get(user_id)

    def get_team(self, team_id: Optional[str]) -> Optional[LiteLLM_TeamTable]:
        if team_id is None:
            return None
        return self.teams.get(team_id)

    def verify_user_password(self, user: LiteLLM_UserTable, password: str) -> bool:
        from litellm.proxy.utils import verify_password

        user_secret = self._secrets.get("users", {}).get(user.user_id, {})
        password_hash = user_secret.get("password_hash")
        if password_hash is None:
            return False
        return verify_password(password, password_hash)

    def set_user_password(self, user_id: str, password: str) -> None:
        from litellm.proxy.utils import hash_password

        self._secrets.setdefault("users", {}).setdefault(user_id, {})[
            "password_hash"
        ] = hash_password(password)
        self._write_secrets()

    def _key_auth_from_secret(self, key_id: str, key_data: dict) -> UserAPIKeyAuth:
        secret_key = key_data.get("key")
        token_hash = key_data.get("token") or (
            hash_token(secret_key) if isinstance(secret_key, str) else key_id
        )
        team = self.get_team(key_data.get("team_id"))
        user = self.get_user(key_data.get("user_id"))
        auth = UserAPIKeyAuth(
            token=token_hash,
            key_name=key_data.get("key_name", key_id),
            key_alias=key_data.get("key_alias", key_id),
            spend=0,
            max_budget=key_data.get("max_budget"),
            expires=_parse_datetime(key_data.get("expires")),
            models=key_data.get("models") or [],
            aliases=key_data.get("aliases") or {},
            config=key_data.get("config") or {},
            user_id=key_data.get("user_id"),
            team_id=key_data.get("team_id"),
            metadata={
                **(key_data.get("metadata") or {}),
                "secret_visible_to_admin": True,
            },
            tpm_limit=key_data.get("tpm_limit"),
            rpm_limit=key_data.get("rpm_limit"),
            allowed_routes=key_data.get("allowed_routes") or [],
            permissions=key_data.get("permissions") or {},
            blocked=key_data.get("blocked", False),
            created_at=_parse_datetime(key_data.get("created_at")),
            updated_at=_parse_datetime(key_data.get("updated_at")),
            user_role=key_data.get("user_role") or (user.user_role if user else None),
            user_email=user.user_email if user else None,
            team_alias=team.team_alias if team else None,
            team_tpm_limit=team.tpm_limit if team else None,
            team_rpm_limit=team.rpm_limit if team else None,
            team_max_budget=team.max_budget if team else None,
            team_soft_budget=team.soft_budget if team else None,
            team_models=team.models if team else [],
            team_blocked=team.blocked if team else False,
            team_metadata=team.metadata if team else None,
        )
        return auth

    def iter_key_auth(self, include_sessions: bool = False) -> List[UserAPIKeyAuth]:
        self.reload()
        key_auths = [
            self._key_auth_from_secret(key_id=key_id, key_data=key_data)
            for key_id, key_data in self._secrets.get("keys", {}).items()
        ]
        if include_sessions:
            key_auths.extend(
                self._key_auth_from_secret(key_id=key_id, key_data=key_data)
                for key_id, key_data in self._secrets.get("ui_sessions", {}).items()
            )
        return key_auths

    def serialize_key_auth(self, key_auth: UserAPIKeyAuth) -> dict:
        key_dict = key_auth.model_dump()
        key_dict["token_id"] = key_auth.token
        for section in ("keys", "ui_sessions"):
            for key_data in self._secrets.get(section, {}).values():
                secret_key = key_data.get("key")
                candidate_hash = key_data.get("token") or (
                    hash_token(secret_key) if isinstance(secret_key, str) else None
                )
                if key_auth.token is not None and candidate_hash == key_auth.token:
                    key_dict["key"] = secret_key
                    return key_dict
        return key_dict

    def get_key_auth_by_hash(self, token_hash: str) -> Optional[UserAPIKeyAuth]:
        self.reload()
        for section in ("keys", "ui_sessions"):
            for key_id, key_data in self._secrets.get(section, {}).items():
                secret_key = key_data.get("key")
                candidate_hash = key_data.get("token") or (
                    hash_token(secret_key) if isinstance(secret_key, str) else key_id
                )
                if secrets.compare_digest(candidate_hash, token_hash):
                    return self._key_auth_from_secret(key_id=key_id, key_data=key_data)
        return None

    def create_ui_session_key(
        self,
        *,
        user_id: str,
        user_role: str,
        user_email: Optional[str],
        team_id: Optional[str] = "litellm-dashboard",
    ) -> str:
        now = datetime.now(timezone.utc)
        expires = now + _duration_to_timedelta(LITELLM_UI_SESSION_DURATION)
        key = f"sk-ui-{uuid.uuid4().hex}"
        token_hash = hash_token(key)
        self._secrets.setdefault("ui_sessions", {})[token_hash] = {
            "key": key,
            "token": token_hash,
            "key_alias": "LiteLLM Admin UI Session",
            "user_id": user_id,
            "user_role": user_role,
            "user_email": user_email,
            "team_id": team_id,
            "models": [],
            "expires": expires.isoformat(),
            "created_at": now.isoformat(),
            "metadata": {"generated_by": "litellm_admin_ui"},
            "max_budget": litellm.max_ui_session_budget,
        }
        self._write_secrets()
        return key

    def upsert_api_key(
        self,
        *,
        key_alias: str,
        key: str,
        user_id: Optional[str],
        team_id: Optional[str],
        models: Optional[List[str]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._secrets.setdefault("keys", {})[key_alias] = {
            "key": key,
            "token": hash_token(key),
            "key_alias": key_alias,
            "user_id": user_id,
            "team_id": team_id,
            "models": models or [],
            "created_at": now,
            "updated_at": now,
        }
        self._write_secrets()

    def delete_api_key(self, key_or_alias: str) -> bool:
        keys = self._secrets.setdefault("keys", {})
        if key_or_alias in keys:
            del keys[key_or_alias]
            self._write_secrets()
            return True
        token_hash = (
            hash_token(key_or_alias) if key_or_alias.startswith("sk-") else key_or_alias
        )
        for alias, key_data in list(keys.items()):
            if key_data.get("token") == token_hash:
                del keys[alias]
                self._write_secrets()
                return True
        return False

    def list_users(
        self,
        *,
        role: Optional[str] = None,
        user_email: Optional[str] = None,
        team: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[LiteLLM_UserTable]:
        users = list(self.users.values())
        if role:
            users = [user for user in users if user.user_role == role]
        if user_email:
            needle = user_email.lower()
            users = [
                user
                for user in users
                if user.user_email and needle in user.user_email.lower()
            ]
        if team:
            users = [user for user in users if team in (user.teams or [])]
        if user_ids:
            id_set = set(user_ids)
            users = [user for user in users if user.user_id in id_set]
        return users

    def list_teams(
        self,
        *,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        team_alias: Optional[str] = None,
    ) -> List[LiteLLM_TeamTable]:
        teams = list(self.teams.values())
        if user_id:
            teams = [
                team
                for team in teams
                if user_id in (team.members or [])
                or user_id in (team.admins or [])
                or any(member.user_id == user_id for member in team.members_with_roles)
            ]
        if team_id:
            teams = [team for team in teams if team.team_id == team_id]
        if team_alias:
            needle = team_alias.lower()
            teams = [
                team
                for team in teams
                if team.team_alias and needle in team.team_alias.lower()
            ]
        return teams

    def team_memberships(self, team_id: str) -> List[LiteLLM_TeamMembership]:
        team = self.get_team(team_id)
        if team is None:
            return []
        return [
            LiteLLM_TeamMembership(
                user_id=member.user_id or "",
                team_id=team_id,
                litellm_budget_table=None,
            )
            for member in team.members_with_roles
        ]

    def user_info(self, user_id: str) -> Tuple[Optional[LiteLLM_UserTable], List, List]:
        user = self.get_user(user_id)
        if user is None:
            return None, [], []
        keys = [
            key
            for key in self.iter_key_auth()
            if key.user_id == user_id or (key.team_id and key.team_id in user.teams)
        ]
        teams = [team for team in self.teams.values() if team.team_id in user.teams]
        return user, keys, teams

    def ensure_default_admin_user(self) -> LiteLLM_UserTable:
        admin = self.get_user(LITELLM_PROXY_ADMIN_NAME)
        if admin is not None:
            return admin
        admin = LiteLLM_UserTable(
            user_id=LITELLM_PROXY_ADMIN_NAME,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            models=[],
            teams=[],
        )
        self.users[admin.user_id] = admin
        return admin
