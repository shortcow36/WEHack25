import streamlit as st


# Page Configuration
st.set_page_config(page_title="Landing Page", page_icon="👋", initial_sidebar_state="collapsed")

# Injecting custom CSS for a linear gradient background
st.markdown("""
    <style>
    /* Apply background gradient to entire body */
    body {
        background: linear-gradient(to right, black, #111827, black);
        color: white;
    }

    /* Ensure the main area also has the gradient */
    .block-container {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the page
st.title("Welcome to Our Awesome App!")

# Redirect to register page if needed
if st.session_state.get("go_register"):
    st.session_state.go_register = False  # Reset the flag
    st.switch_page("pages/register.py")  # Navigate to the register page

# Register Button
if st.button("Register"):
    st.session_state.go_register = True  # Set the flag to true
    st.rerun()  # Trigger a rerun to navigate to the register page
