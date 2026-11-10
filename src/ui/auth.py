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
    _render_login_portal()
    return False


_LOGIN_CSS = """
<style>
/* ---- Page frame: no sidebar / toolbar, branded backdrop ---- */
section[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stApp"] {
    background:
        radial-gradient(1100px 520px at 12% -10%, rgba(34, 197, 94, 0.16), transparent 60%),
        radial-gradient(900px 480px at 100% 110%, rgba(20, 184, 166, 0.14), transparent 60%),
        #0a0f1a !important;
}
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 7vh !important; padding-bottom: 4vh !important; max-width: 1040px !important;
}

/* ---- The card ---- */
.st-key-login_shell {
    background: #0f1726;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 22px;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(34, 197, 94, 0.05);
    overflow: hidden;
    padding: 0 !important;
    gap: 0 !important;
}
.st-key-login_shell [data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: stretch !important; }
.st-key-login_shell [data-testid="stColumn"] { padding: 44px 46px 36px 46px !important; }
.st-key-login_shell [data-testid="stColumn"]:first-child {
    background:
        radial-gradient(420px 260px at 0% 0%, rgba(255, 255, 255, 0.10), transparent 70%),
        linear-gradient(155deg, #166534 0%, #0f766e 58%, #115e59 100%);
}

/* ---- Brand panel ---- */
.lp-brand { color: #ecfdf5; display: flex; flex-direction: column; height: 100%; min-height: 400px; }
.lp-logo {
    width: 54px; height: 54px; border-radius: 15px; display: flex; align-items: center; justify-content: center;
    font-size: 1.75rem; background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.22);
}
.lp-brand h1 { color: #ffffff; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.6px; line-height: 1.15; margin: 22px 0 10px 0; padding: 0; }
.lp-tag { color: rgba(236, 253, 245, 0.86); font-size: 0.98rem; line-height: 1.55; margin: 0 0 28px 0; }
.lp-feat { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 16px; }
.lp-ico {
    flex: 0 0 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    background: rgba(255, 255, 255, 0.13); font-size: 1.0rem;
}
.lp-feat b { display: block; color: #ffffff; font-size: 0.92rem; font-weight: 700; }
.lp-feat span { color: rgba(236, 253, 245, 0.78); font-size: 0.82rem; line-height: 1.4; }
.lp-foot { margin-top: auto; padding-top: 22px; font-size: 0.76rem; color: rgba(236, 253, 245, 0.66); letter-spacing: 0.2px; }

/* ---- Form panel ---- */
.lp-form-head { margin-bottom: 26px; }
.lp-chip {
    display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px;
    text-transform: uppercase; color: #4ade80; background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.28); padding: 4px 10px; border-radius: 999px;
}
.lp-form-head h2 { color: #f8fafc; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.4px; margin: 14px 0 6px 0; padding: 0; }
.lp-form-head p { color: #94a3b8; font-size: 0.92rem; margin: 0; }
.st-key-login_shell div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.st-key-login_shell [data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-size: 0.84rem !important; font-weight: 600 !important; }
.st-key-login_shell [data-baseweb="input"], .st-key-login_shell [data-baseweb="base-input"] {
    background: #0a1120 !important; border-radius: 11px !important;
}
.st-key-login_shell [data-baseweb="input"] {
    border: 1px solid rgba(148, 163, 184, 0.22) !important; transition: border-color .15s ease, box-shadow .15s ease;
}
.st-key-login_shell [data-baseweb="input"]:focus-within {
    border-color: #22c55e !important; box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18) !important;
}
.st-key-login_shell input { color: #f1f5f9 !important; caret-color: #22c55e !important; height: 46px; font-size: 0.95rem !important; }
.st-key-login_shell input::placeholder { color: #64748b !important; }
.st-key-login_shell button[kind="primaryFormSubmit"], .st-key-login_shell [data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important; border: none !important;
    color: #04210f !important; font-weight: 800 !important; height: 48px; border-radius: 11px !important;
    margin-top: 8px; box-shadow: 0 10px 24px rgba(22, 163, 74, 0.28); transition: transform .12s ease, box-shadow .12s ease;
}
.st-key-login_shell button[kind="primaryFormSubmit"]:hover, .st-key-login_shell [data-testid="stBaseButton-primaryFormSubmit"]:hover {
    transform: translateY(-1px); box-shadow: 0 14px 30px rgba(22, 163, 74, 0.38);
}
.st-key-login_shell button[kind="primaryFormSubmit"] p { color: #04210f !important; font-weight: 800 !important; font-size: 0.98rem !important; }
.lp-note { display: flex; gap: 8px; align-items: center; color: #64748b; font-size: 0.78rem; margin-top: 22px; padding-top: 18px; border-top: 1px solid rgba(148, 163, 184, 0.12); }
.lp-below { text-align: center; color: #475569; font-size: 0.76rem; margin-top: 22px; }

@media (max-width: 640px) {
    .st-key-login_shell [data-testid="stColumn"] { padding: 30px 24px 26px 24px !important; }
    .lp-brand { min-height: 0; }
    .lp-feats, .lp-foot { display: none; }
    .lp-brand h1 { font-size: 1.45rem; margin-top: 14px; }
    .lp-tag { margin-bottom: 0; }
}
</style>
"""

_LOGIN_BRAND_HTML = """
<div class="lp-brand">
    <div class="lp-logo">♻️</div>
    <h1>Coripet RVM<br>Predictor</h1>
    <p class="lp-tag">Decide where every reverse vending machine pays off — with forecasts trained on the real Coripet network.</p>
    <div class="lp-feats">
        <div class="lp-feat"><div class="lp-ico">📈</div><div><b>Volume forecasts</b><span>Daily, monthly and yearly bottles and users for any Italian store.</span></div></div>
        <div class="lp-feat"><div class="lp-ico">🛡️</div><div><b>Go / No-Go clearance</b><span>Six objective gates on cannibalization, breakeven, format and logistics.</span></div></div>
        <div class="lp-feat"><div class="lp-ico">🔍</div><div><b>Fleet monitoring</b><span>Outliers, defects and performance drops across installed machines.</span></div></div>
    </div>
    <div class="lp-foot">Coripet Consorzio Volontario per il Riciclo del PET</div>
</div>
"""


def _render_login_portal():
    """Two-panel branded sign-in card (brand panel + credential form)."""
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    login_placeholder = st.empty()
    with login_placeholder.container():
        with st.container(key="login_shell"):
            col_brand, col_form = st.columns([1.0, 1.05], gap="small")
            with col_brand:
                st.html(_LOGIN_BRAND_HTML)

            with col_form:
                st.html(
                    """<div class="lp-form-head">
                        <span class="lp-chip">🔒 Secure access</span>
                        <h2>Welcome back</h2>
                        <p>Sign in to the decision support console.</p>
                    </div>"""
                )

                with st.form("admin_login_form", clear_on_submit=False, border=False):
                    user_input = st.text_input(
                        "Username",
                        placeholder="Enter your username",
                        key="login_user_field",
                        autocomplete="username",
                    ).strip()
                    pass_input = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="login_pass_field",
                        autocomplete="current-password",
                    )
                    submitted = st.form_submit_button("Sign in  →", type="primary", use_container_width=True)

                    if submitted:
                        if check_credentials(user_input, pass_input):
                            login_placeholder.empty()
                            token = generate_auth_token(user_input)
                            _set_auth_param(token)
                            st.session_state["is_authenticated"] = True
                            st.session_state["authenticated_user"] = user_input
                            st.toast("✅ Authentication successful. Welcome to Coripet Predictor!", icon="🔓")
                            st.rerun()
                        elif not user_input or not pass_input:
                            st.error("Please enter both your username and password.")
                        else:
                            st.error("Invalid username or password. Please try again.")

                st.html(
                    """<div class="lp-note">🛡️ <span>Authorized personnel only · your session stays signed in for 60 days on this browser.</span></div>"""
                )

        st.html("""<div class="lp-below">Coripet Intelligent Decision Support System</div>""")
