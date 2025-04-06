
import streamlit as st

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

# Title and instructions
st.title("User Profile")

# Property Owner Toggle
property_owner = st.radio("Are you a Property Owner?", ["Yes", "No"])

# Profile Details Section
st.header("Profile Details")
first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
contact_email = st.text_input("Contact Email")
contact_number = st.text_input("Contact Number")
city = st.text_input("City")

# State dropdown with search feature
states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", 
          "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
          "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", 
          "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
          "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
          "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

# Set session state for state before selecting the dropdown
if 'state' not in st.session_state:
    st.session_state.state = states[0]  # Default to first state if not set

state = st.selectbox("State", states, index=states.index(st.session_state.state), key="state")

# Operational Fit Section
st.header("Operational Fit")
business_hours_start = st.selectbox("Business Hours (Start)", ["6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"])
business_hours_end = st.selectbox("Business Hours (End)", ["6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM", "11:00 PM"])

business_type = st.selectbox("Business Type", ["Retail / Consumer", "Food / Beverage", "Service Based", "Creative or Education"])

noise_level = st.selectbox("Noise Level", ["Low", "Medium", "High"])

# Conditional Property Owner fields
if property_owner == "Yes":
    property_address = st.text_input("Property Address")
    square_footage = st.number_input("Square Footage of Property", min_value=0)

# Save Profile button
if st.button("Save Profile"):
    # Store user data in session state
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name
    st.session_state.contact_email = contact_email
    st.session_state.contact_number = contact_number
    st.session_state.city = city
    st.session_state.state = state
    st.session_state.business_hours_start = business_hours_start
    st.session_state.business_hours_end = business_hours_end
    st.session_state.business_type = business_type
    st.session_state.noise_level = noise_level

    if property_owner == "Yes":
        st.session_state.property_owner = True
        st.session_state.property_address = property_address
        st.session_state.square_footage = square_footage
    else:
        st.session_state.property_owner = False

    # Set flags to navigate after saving profile
    if st.session_state.property_owner:
        st.session_state.go_to_risk_analysis = True
    else:
        st.session_state.go_to_user_matches = True

# Conditional redirection after Save Profile
if st.session_state.get("go_to_risk_analysis"):
    st.session_state.go_to_risk_analysis = False
    st.switch_page("pages/risk_analysis.py")  # Navigate to Risk Analysis page

if st.session_state.get("go_to_user_matches"):
    st.session_state.go_to_user_matches = False
    st.switch_page("pages/users_matched.py")  # Navigate to User Matches page

# Display Buttons at the Bottom
col1, col2 = st.columns([1, 1])  # Create two columns for the buttons
with col1:
    if property_owner == "Yes":
        if st.button("Risk Score"):  # Only show if the user is a property owner
            st.session_state.go_to_risk_analysis = True
            st.switch_page("pages/risk_analysis.py")  # Navigate to risk analysis page
with col2:
    if st.button("Matches"):  # This button is shown for all users
        st.session_state.go_to_user_matches = True
        st.switch_page("pages/users_matched.py")  # Navigate to user matches page
