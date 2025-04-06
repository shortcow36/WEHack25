

import streamlit as st


# Page Configuration
st.set_page_config(page_title="Register Page", page_icon="📝", initial_sidebar_state="collapsed")

# Inject custom CSS to change the background color of the app container
# Inject custom CSS to change the background color of the app container
st.markdown("""
    <style>
    /* Target the header container using stAppHeader */
    .stAppHeader {
        background-color: black;  /* Set the background color to black */
        color: white;  /* Set text color to white for contrast */
    }
    /* Target the app view container directly */
    .stAppViewContainer {
        background: linear-gradient(to bottom, black, #111827, black);
        color: white;
        min-height: 100vh;  /* Ensure the background takes full height */
    }
    h1#co-sqrd {
        font-weight: 400;
    }
    h3#connect-to-explore-co-venture-matches-and-real-estate-insights {
        font-weight: 300; /* Sets the font weight to 400 */
    }
    /* Optional: Style the main content area */
    .main {
        color: white;  /* Change text color for contrast */
    }
    </style>
""", unsafe_allow_html=True)



# Title
st.title("Co^SQRD")

# Smaller caption
st.markdown("### Sign Up for your Account.")

# caption
st.markdown("### Connect to explore co-venture matches and real estate insights.")


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
    st.switch_page("pages/profile.py")  # Navigate to the profile pages





