

import streamlit as st


# Page Configuration
st.set_page_config(page_title="Register Page", page_icon="📝", initial_sidebar_state="collapsed")

# Injecting custom CSS for linear gradient background
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, black, #111827, black);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the page
st.title("Sign Up for Our App")

# Input fields for registration
first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

# Registration logic
if st.button("Register"):
    # Basic validation: check if all fields are filled
    if first_name and last_name and email and password and confirm_password:
        # Ensure passwords match
        if password == confirm_password:
            # Set the session state to mark registration as successful
            st.session_state.registered = True
            st.session_state.user_name = first_name  # Store the first name (or other user info)
            
            # Redirect to the profile page after successful registration
            st.session_state.go_profile = True
            st.rerun()  # Trigger rerun to switch to the profile page
        else:
            st.error("Passwords do not match!")
    else:
        st.error("Please fill in all fields.")

# Redirect to profile page
if st.session_state.get("go_profile"):
    st.session_state.go_profile = False  # Reset the flag
    st.switch_page("pages/profile.py")  # Navigate to the profile page


