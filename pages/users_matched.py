import streamlit as st
import pandas as pd

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



st.title("Your Co-Venturer Matches")


# Hardcoded data
data = {
    "Name": ["John Doe", "Jane Smith", "Mark Johnson", "Emily Davis", "Michael Brown"],
    "Email": ["john.doe@email.com", "jane.smith@email.com", "mark.johnson@email.com", "emily.davis@email.com", "michael.brown@email.com"],
    "Contact": ["123-456-7890", "234-567-8901", "345-678-9012", "456-789-0123", "567-890-1234"],
    "Location": ["Dallas, TX", "Fort Worth, TX", "Arlington, TX", "Plano, TX", "Irving, TX"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display each user in a clickable box
for idx, row in df.iterrows():
    with st.expander(f"{row['Name']}"):
        st.write(f"**Email:** {row['Email']}")
        st.write(f"**Contact:** {row['Contact']}")
        st.write(f"**Location:** {row['Location']}")
        st.write(f"**More Information:** Here you can add more details about the user.")


