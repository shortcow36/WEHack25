'''import streamlit as st

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
    /* Optional: Style the main content area */
    .main {
        color: white;  /* Change text color for contrast */
    }
    </style>
""", unsafe_allow_html=True)



st.title("Risk Score:")
# Add content for the risk analysis page here.
st.write("This is where the risk analysis will be displayed.")
'''


import streamlit as st
from Linear_Regression import predict_risk_score  # adjust path if needed

st.set_page_config(page_title="Risk Score", page_icon="📈")

st.title("Predicted Risk Score")

# Check if lat/lon were passed
if "lat" in st.session_state and "lon" in st.session_state:
    lat = st.session_state.lat
    lon = st.session_state.lon
    st.write(f"📍 Coordinates: {lat}, {lon}")

    # Call your backend prediction function
    risk_score = predict_risk_score(lat, lon)

    # Display the result
    st.metric(label="Estimated Risk Score", value=f"{risk_score:.2f}")
else:
    st.warning("Please go to the Home page and enter coordinates.")
    st.page_link("Home.py", label="🏠 Go to Home Page")
