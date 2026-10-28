"""
Administrative Authentication Module for Coripet RVM Predictor.
Implements timing-attack resistant credentials validation,
session state management, and protected administrative gating.
"""

import os
import hmac
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Configured admin credentials with secure defaults
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin").strip()
DEFAULT_ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "CoripetAdmin2026*").strip()


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


def logout():
    """Clears authentication session state and forces a fresh rerun."""
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
    Complete security gate for the application.
    If authenticated, renders user badge in the sidebar and returns True.
    If unauthenticated, hides all internal components, renders the login portal,
    and returns False.
    """
    if st.session_state.get("is_authenticated", False):
        render_sidebar_user_badge()
        return True

    # Hide sidebar completely while logged out
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .login-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 32px 36px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
            margin-top: 40px;
        }
        [data-theme="light"] .login-card {
            background: #ffffff;
            border: 1px solid rgba(0, 0, 0, 0.10);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Centered Login Card
    col_l, col_center, col_r = st.columns([1.5, 3.0, 1.5])
    with col_center:
        st.markdown(
            """
            <div class="login-card">
                <div style="text-align: center; margin-bottom: 24px;">
                    <div style="font-size: 2.8rem; margin-bottom: 6px;">♻️</div>
                    <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.4px;">
                        Coripet RVM Predictor
                    </h2>
                    <p style="margin: 6px 0 0 0; opacity: 0.75; font-size: 0.88rem;">
                        🔒 Secure Administrative Access Portal
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("admin_login_form", clear_on_submit=False):
            st.markdown("<p style='margin: 0 0 4px 0; font-weight: 600; font-size: 0.85rem;'>Username</p>", unsafe_allow_html=True)
            user_input = st.text_input(
                "Username",
                placeholder="Enter admin username...",
                key="login_user_field",
                label_visibility="collapsed"
            ).strip()

            st.markdown("<p style='margin: 12px 0 4px 0; font-weight: 600; font-size: 0.85rem;'>Password</p>", unsafe_allow_html=True)
            pass_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter admin password...",
                key="login_pass_field",
                label_visibility="collapsed"
            )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔐 Sign In to Console",
                type="primary",
                use_container_width=True
            )

            if submitted:
                if check_credentials(user_input, pass_input):
                    st.session_state["is_authenticated"] = True
                    st.session_state["authenticated_user"] = user_input
                    st.toast("✅ Authentication successful. Welcome to Coripet Predictor!", icon="🔓")
                    st.rerun()
                else:
                    st.error("❌ Authentication failed: Invalid username or password.")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 18px; font-size: 0.78rem; opacity: 0.60;">
                Coripet Intelligent Decision Support System &bull; Authorized Personnel Only
            </div>
            """,
            unsafe_allow_html=True
        )

    return False

