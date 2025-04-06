# import streamlit as st


# # Page Configuration
# st.set_page_config(page_title="Landing Page", page_icon="👋", initial_sidebar_state="collapsed")

# # Inject custom CSS to change the background color of the app container
# st.markdown("""
#     <style>

    
#     /* Target the header container using stAppHeader */
#     .stAppHeader {
#         background-color: black;  /* Set the background color to black */
#         color: white;  /* Set text color to white for contrast */
#     }
#     /* Target the app view container directly */
#     .stAppViewContainer {
#         background: linear-gradient(to bottom, black, #111827, black);
#         color: white;
#         min-height: 100vh;  /* Ensure the background takes full height */
#     }

#     /* Optional: Style the main content area */
#     .main {
#         color: white;  /* Change text color for contrast */
#     }

#     /* Style the title: Center and change font to Inter */
#     .stMainBlockContainer {
#         text-align: center;  /* Center the title */
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Title of the page
# st.title("Co^SQRD")

# # Redirect to register page if needed
# if st.session_state.get("go_register"):
#     st.session_state.go_register = False  # Reset the flag
#     st.switch_page("pages/register.py")  # Navigate to the register page

# # Register Button
# if st.button("Register"):
#     st.session_state.go_register = True  # Set the flag to true
#     st.rerun()  # Trigger a rerun to navigate to the register page


import streamlit as st

# Page Configuration
st.set_page_config(page_title="Landing Page", page_icon="👋", initial_sidebar_state="collapsed")

# Inject custom CSS
st.markdown("""
    <style>
    .stAppHeader {
        background-color: black;
        color: white;
    }
    .stAppViewContainer {
        background: linear-gradient(to bottom, black, #111827, black);
        color: white;
        min-height: 100vh;
    }
    h1#co-sqrd {
        font-weight: 400;
    }
    h3#co-venturing-and-risk-analysis-made-easy {
        font-weight: 200; /* Or 500 or 300 */
    }
    .main {
        color: white;
    }
    .stMainBlockContainer {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Co^SQRD")

# Slogan
st.markdown("### Co-venturing and risk analysis made easy.")

# Redirect to register page if needed
if st.session_state.get("go_register"):
    st.session_state.go_register = False
    st.switch_page("pages/register.py")

# Centered buttons stacked vertically
col_center = st.container()
with col_center:
    if st.button("Register"):
        st.session_state.go_register = True
        st.rerun()
    st.button("Sign In")  # Placeholder for future functionality





