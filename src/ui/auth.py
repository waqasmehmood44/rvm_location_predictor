"""
Administrative Authentication Module for Coripet RVM Predictor.
Implements timing-attack resistant credentials validation,
session state management, protected administrative gating,
and persistent authenticated sessions across browser refreshes
via dual-layer HMAC-signed tokens and browser localStorage.
"""

import os
import time
import hmac
import hashlib
from typing import Optional
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Configured admin credentials with secure defaults
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin").strip()
DEFAULT_ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "CoripetAdmin2026*").strip()
AUTH_SALT = os.getenv("AUTH_SALT", "coripet_auth_token_salt_2026")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def check_credentials(username: str, password: str) -> bool:
    """
    Validates username and password using constant-time comparison
    to prevent timing side-channel attacks.
    """
    if not username or not password:
        return False

    target_user = os.getenv("ADMIN_USERNAME", DEFAULT_ADMIN_USER).strip()
    target_pass = os.getenv("ADMIN_PASSWORD", DEFAULT_ADMIN_PASS).strip()

    user_match = hmac.compare_digest(username.strip().encode("utf-8"), target_user.encode("utf-8"))
    pass_match = hmac.compare_digest(password.encode("utf-8"), target_pass.encode("utf-8"))

    return user_match and pass_match


def generate_auth_token(username: str) -> str:
    """Generates a cryptographically signed authentication token."""
    target_pass = os.getenv("ADMIN_PASSWORD", DEFAULT_ADMIN_PASS).strip()
    timestamp = int(time.time())
    secret = (target_pass + AUTH_SALT).encode("utf-8")
    payload = f"{username}:{timestamp}"
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_auth_token(token: str, max_age_days: int = 60) -> Optional[str]:
    """
    Verifies the HMAC signature and timestamp of an authentication token.
    Returns the authenticated username if valid, otherwise None.
    Default validity: 60 days.
    """
    if not token or ":" not in token:
        return None
    parts = token.split(":")
    if len(parts) != 3:
        return None
    username, timestamp_str, sig = parts
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return None

    # Expiry verification
    if time.time() - timestamp > max_age_days * 86400:
        return None

    target_user = os.getenv("ADMIN_USERNAME", DEFAULT_ADMIN_USER).strip()
    target_pass = os.getenv("ADMIN_PASSWORD", DEFAULT_ADMIN_PASS).strip()

    secret = (target_pass + AUTH_SALT).encode("utf-8")
    expected_sig = hmac.new(secret, f"{username}:{timestamp}".encode("utf-8"), hashlib.sha256).hexdigest()

    if hmac.compare_digest(sig, expected_sig) and username == target_user:
        return username
    return None


def _get_auth_param() -> Optional[str]:
    """Retrieves auth token from URL query parameters across Streamlit versions."""
    try:
        if hasattr(st, "query_params"):
            val = st.query_params.get("auth")
            if isinstance(val, list):
                return val[0] if val else None
            return val
        elif hasattr(st, "experimental_get_query_params"):
            params = st.experimental_get_query_params()
            vals = params.get("auth", [])
            return vals[0] if vals else None
    except Exception:
        pass
    return None


def _set_auth_param(token: str):
    """Sets persistent auth token in URL query parameters."""
    try:
        if hasattr(st, "query_params"):
            st.query_params["auth"] = token
        elif hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params(auth=token)
    except Exception:
        pass


def _clear_auth_param():
    """Removes auth token from URL query parameters."""
    try:
        if hasattr(st, "query_params"):
            if "auth" in st.query_params:
                del st.query_params["auth"]
        elif hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params()
    except Exception:
        pass


def logout():
    """Clears authentication session state, URL token, and forces a fresh rerun."""
    _clear_auth_param()
    st.session_state["is_authenticated"] = False
    st.session_state.pop("authenticated_user", None)
    st.rerun()


def render_sidebar_user_badge():
    """Renders user badge and sign-out button in the sidebar."""
    username = st.session_state.get("authenticated_user", "admin")
    with st.sidebar:
        st.markdown(
            f"""
            <div style="background: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.72rem; color: #16a34a; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Logged In</div>
                    <div style="font-size: 0.90rem; font-weight: 700; color: inherit;">👤 {username}</div>
                </div>
                <div>
                    <span style="font-size: 0.70rem; background: rgba(34, 197, 94, 0.20); color: #16a34a; padding: 3px 8px; border-radius: 4px; font-weight: 800;">ADMIN</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🚪 Sign Out", key="btn_auth_signout", use_container_width=True, help="End session and return to login screen"):
            logout()
        st.markdown("<div style='border-bottom: 1px solid rgba(128, 128, 128, 0.20); margin: 10px 0 14px 0;'></div>", unsafe_allow_html=True)


def require_auth() -> bool:
    """
    Security gate for the application using native Streamlit session state and URL tokens.
    Eliminates fragile iframe location.replace calls and ghost form retention.
    """
    if not AUTH_ENABLED:
        return True

    # 1. Active session in current memory
    if st.session_state.get("is_authenticated", False):
        render_sidebar_user_badge()
        return True

    # 2. Page refresh check: persistent token from URL query parameters
    token = _get_auth_param()
    if token:
        valid_user = verify_auth_token(token)
        if valid_user:
            st.session_state["is_authenticated"] = True
            st.session_state["authenticated_user"] = valid_user
            render_sidebar_user_badge()
            return True
        else:
            _clear_auth_param()

    # 3. Unauthenticated: Render Login Portal
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .login-unified-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 32px 36px 28px 36px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.50);
            margin-top: 35px;
        }
        [data-theme="light"] .login-unified-card {
            background: #ffffff;
            border: 1px solid rgba(0, 0, 0, 0.10);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
        }
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }
        div[data-testid="stTextInput"] input {
            caret-color: #22c55e !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_l, col_center, col_r = st.columns([1.2, 2.6, 1.2])
    with col_center:
        st.markdown(
            """
            <div class="login-unified-card">
                <div style="text-align: center; margin-bottom: 22px;">
                    <div style="font-size: 2.8rem; margin-bottom: 6px;">♻️</div>
                    <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.4px; color: inherit;">
                        Coripet RVM Predictor
                    </h2>
                    <p style="margin: 6px 0 0 0; opacity: 0.70; font-size: 0.88rem; font-weight: 500;">
                        🔒 Secure Administrative Access Portal
                    </p>
                </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("admin_login_form", clear_on_submit=False, border=False):
            st.markdown("<p style='margin: 0 0 4px 0; font-weight: 600; font-size: 0.85rem; opacity: 0.85;'>Username</p>", unsafe_allow_html=True)
            user_input = st.text_input(
                "Username",
                placeholder="Enter admin username...",
                key="login_user_field",
                label_visibility="collapsed"
            ).strip()

            st.markdown("<p style='margin: 12px 0 4px 0; font-weight: 600; font-size: 0.85rem; opacity: 0.85;'>Password</p>", unsafe_allow_html=True)
            pass_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter admin password...",
                key="login_pass_field",
                label_visibility="collapsed"
            )

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔐 Sign In to Console",
                type="primary",
                use_container_width=True
            )

            if submitted:
                if check_credentials(user_input, pass_input):
                    token = generate_auth_token(user_input)
                    _set_auth_param(token)
                    st.session_state["is_authenticated"] = True
                    st.session_state["authenticated_user"] = user_input
                    st.toast("✅ Authentication successful. Welcome to Coripet Predictor!", icon="🔓")
                    st.rerun()
                else:
                    st.error("❌ Authentication failed: Invalid username or password.")

        st.markdown(
            """
                <div style="text-align: center; margin-top: 20px; font-size: 0.76rem; opacity: 0.55;">
                    Coripet Intelligent Decision Support System &bull; Authorized Personnel Only
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    return False
