import streamlit as st

# Injecting custom CSS for linear gradient background
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, black, #111827, black);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sample states list for dropdown (abbreviated, can be extended as needed)
states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", 
          "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
          "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", 
          "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
          "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
          "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

# Title and instructions
st.title("User Profile")

# Property Owner Toggle
property_owner = st.radio("Property Owner?", ["Yes", "No"])

# Profile Details Section
st.header("Profile Details")
first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
contact_email = st.text_input("Contact Email")
contact_number = st.text_input("Contact Number")
city = st.text_input("City")

# State dropdown with search feature
state = st.selectbox("State", states, index=0, key="state")

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

# Button to submit (later to save data to MongoDB)
if st.button("Save Profile"):
    # Print for debugging (replace with MongoDB saving code later)
    st.write("Profile Saved:")
    st.write(f"Name: {first_name} {last_name}")
    st.write(f"Email: {contact_email}")
    st.write(f"Contact Number: {contact_number}")
    st.write(f"City: {city}, {state}")
    st.write(f"Business Hours: {business_hours_start} to {business_hours_end}")
    st.write(f"Business Type: {business_type}")
    st.write(f"Noise Level: {noise_level}")
    
    if property_owner == "Yes":
        st.write(f"Property Address: {property_address}")
        st.write(f"Square Footage: {square_footage}")


