from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path

import streamlit as st


AUTH_SESSION_KEY = "app_authenticated"
AUTH_USER_SESSION_KEY = "app_authenticated_user"
AUTH_MODE_SESSION_KEY = "auth_mode"
LOCAL_DEV_USERNAME = "admin"
LOCAL_DEV_PASSWORD = "risklab"
PBKDF2_ITERATIONS = 210_000
USERS_DB_PATH = Path("data") / "risklab.db"


def get_auth_credentials() -> tuple[str, str]:
    """Return configured app credentials, falling back to local development values."""
    username, password, _ = _get_configured_credentials()
    return username, password


def is_authenticated() -> bool:
    return bool(st.session_state.get(AUTH_SESSION_KEY, False))


def render_login() -> None:
    _inject_auth_styles()
    st.session_state.setdefault(AUTH_MODE_SESSION_KEY, "login")

    _, content, _ = st.columns([1, 1.15, 1])
    with content:
        st.markdown('<div class="risklab-auth-card-shell">', unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state[AUTH_MODE_SESSION_KEY] == "register":
                _render_registration_form()
            else:
                _render_login_form()
        st.markdown("</div>", unsafe_allow_html=True)


def _inject_auth_styles() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(36, 99, 235, 0.14), transparent 30rem),
                    linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            }

            .risklab-auth-card-shell {
                margin: 3.5rem auto 1rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 24px;
                box-shadow: 0 24px 70px rgba(15, 23, 42, 0.12);
                padding: 1.5rem 1.65rem 1.25rem;
            }

            .risklab-auth-title {
                color: #0f172a;
                font-size: 2.25rem;
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.1;
                margin: 0 0 0.35rem;
                text-align: center;
            }

            .risklab-auth-copy {
                color: #64748b;
                font-size: 0.98rem;
                line-height: 1.45;
                margin: 0 0 1.6rem;
                text-align: center;
            }

            .risklab-auth-separator {
                align-items: center;
                color: #94a3b8;
                display: flex;
                font-size: 0.82rem;
                font-weight: 600;
                gap: 0.75rem;
                margin: 1.25rem 0 1rem;
                text-transform: uppercase;
            }

            .risklab-auth-separator::before,
            .risklab-auth-separator::after {
                background: #e2e8f0;
                content: "";
                flex: 1;
                height: 1px;
            }

            div[data-testid="stForm"] {
                border: 0;
                padding: 0;
            }

            .stTextInput label {
                color: #334155;
                font-weight: 700;
            }

            .stTextInput input {
                border-radius: 14px;
                min-height: 3rem;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                border-radius: 14px;
                font-weight: 800;
                min-height: 3rem;
            }

            @media (max-width: 640px) {
                .risklab-auth-card-shell {
                    margin-top: 1.5rem;
                }

                [data-testid="stVerticalBlockBorderWrapper"] {
                    border-radius: 18px;
                    padding: 1rem 0.85rem;
                }

                .risklab-auth-title {
                    font-size: 1.9rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_auth_mode(mode: str) -> None:
    st.session_state[AUTH_MODE_SESSION_KEY] = mode


def _render_auth_header(copy: str) -> None:
    st.markdown(
        f"""
        <div class="risklab-auth-header">
            <h1 class="risklab-auth-title">RiskLab</h1>
            <p class="risklab-auth-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_login() -> None:
    if is_authenticated():
        return

    render_login()
    st.stop()


def logout_button() -> None:
    if not is_authenticated():
        return

    if st.sidebar.button("Cerrar sesion", use_container_width=True):
        st.session_state.pop(AUTH_SESSION_KEY, None)
        st.session_state.pop(AUTH_USER_SESSION_KEY, None)
        _rerun()


def _render_login_form() -> None:
    _render_auth_header("Accede a tu panel de riesgo con tu usuario y contrasena.")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario")
        password = st.text_input("Contrasena", type="password")
        submitted = st.form_submit_button("Iniciar sesion", type="primary", use_container_width=True)

    st.markdown('<div class="risklab-auth-separator">o</div>', unsafe_allow_html=True)
    st.button(
        "Crear cuenta nueva",
        use_container_width=True,
        on_click=_set_auth_mode,
        args=("register",),
    )

    if not submitted:
        return

    success = _authenticate_user(username.strip(), password)
    if success:
        st.session_state[AUTH_SESSION_KEY] = True
        st.session_state[AUTH_USER_SESSION_KEY] = username.strip()
        _rerun()

    st.error("Usuario o contrasena incorrectos. Revisa los datos e intentalo de nuevo.")


def _render_registration_form() -> None:
    _render_auth_header("Crea una cuenta para guardar el acceso al dashboard.")
    with st.form("registration_form", clear_on_submit=False):
        username = st.text_input("Usuario")
        password = st.text_input("Contrasena", type="password")
        confirm_password = st.text_input("Confirmar contrasena", type="password")
        submitted = st.form_submit_button("Crear cuenta", type="primary", use_container_width=True)

    st.button(
        "Ya tengo cuenta / Iniciar sesion",
        use_container_width=True,
        on_click=_set_auth_mode,
        args=("login",),
    )

    if not submitted:
        return

    ok, message = _register_user(username.strip(), password, confirm_password)
    if not ok:
        st.error(message)
        return

    st.success("Cuenta creada correctamente. Ya puedes iniciar sesion.")


def _register_user(username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    if not username:
        return False, "Ingresa un usuario."
    if not password:
        return False, "Ingresa una contrasena."
    if password != confirm_password:
        return False, "La contrasena y la confirmacion no coinciden."

    try:
        _ensure_users_table()
        if _user_exists(username):
            return False, "Ese usuario ya existe. Elige otro nombre de usuario."

        salt = secrets.token_bytes(16)
        password_hash = _hash_password(password, salt)
        with _connect_users_db() as connection:
            connection.execute(
                """
                INSERT INTO app_users (username, salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (username, salt.hex(), password_hash.hex()),
            )
            connection.commit()
    except sqlite3.Error:
        return False, "No fue posible crear la cuenta en este momento. Intentalo de nuevo."

    return True, "Cuenta creada correctamente."


def _authenticate_user(username: str, password: str) -> bool:
    if not username or not password:
        return False

    try:
        _ensure_users_table()
        user_record = _get_user_record(username)
        if user_record is not None:
            salt_hex, password_hash_hex = user_record
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(password_hash_hex)
            return hmac.compare_digest(_hash_password(password, salt), expected_hash)

        return _configured_credentials_match(username, password)
    except (ValueError, sqlite3.Error):
        return False


def _configured_credentials_match(username: str, password: str) -> bool:
    expected_username, expected_password, source = _get_configured_credentials()
    if source == "local" and _has_registered_users():
        return False
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(
        password,
        expected_password,
    )


def _get_configured_credentials() -> tuple[str, str, str]:
    try:
        username = st.secrets["APP_USERNAME"]
        password = st.secrets["APP_PASSWORD"]
        return str(username), str(password), "secrets"
    except (FileNotFoundError, KeyError):
        return LOCAL_DEV_USERNAME, LOCAL_DEV_PASSWORD, "local"


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )


def _connect_users_db() -> sqlite3.Connection:
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(USERS_DB_PATH)


def _ensure_users_table() -> None:
    with _connect_users_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def _has_registered_users() -> bool:
    try:
        _ensure_users_table()
        with _connect_users_db() as connection:
            cursor = connection.execute("SELECT 1 FROM app_users LIMIT 1")
            return cursor.fetchone() is not None
    except sqlite3.Error:
        return False


def _user_exists(username: str) -> bool:
    return _get_user_record(username) is not None


def _get_user_record(username: str) -> tuple[str, str] | None:
    with _connect_users_db() as connection:
        cursor = connection.execute(
            "SELECT salt, password_hash FROM app_users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()

    if row is None:
        return None
    return str(row[0]), str(row[1])


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:
        st.experimental_rerun()
