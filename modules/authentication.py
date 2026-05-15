import streamlit as st


# ---------- LOGIN FUNCTION ----------

def login():

    st.markdown(
        """
        <div class="login-container">
            <h1 class="login-title">🤖 AI Recruitment Platform</h1>
            <p class="login-subtitle">
                Secure Recruiter Login Portal
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    email = st.text_input(
        "📧 Recruiter Email"
    )

    password = st.text_input(
        "🔒 Password",
        type="password"
    )

    login_button = st.button(
        "🚀 Login"
    )

    if login_button:

        # Demo Credentials

        if (
            email == "bittu682001@gmail.com"
            and
            password == "bittu123"
        ):

            st.session_state.logged_in = True

            st.success(
                "Login Successful!"
            )

            st.rerun()

        else:

            st.error(
                "Invalid Email or Password"
            )


# ---------- LOGOUT FUNCTION ----------

def logout():

    if st.sidebar.button("🚪 Logout"):

        st.session_state.logged_in = False

        st.rerun()