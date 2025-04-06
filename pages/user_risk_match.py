'''import streamlit as st
from pymongo import MongoClient
import pandas as pd

# Set your MongoDB URI (replace with your actual connection string)
MONGO_URI = "mongodb+srv://roshnibeddhannan:Kiki@wehack-cluster.i8q9nhv.mongodb.net/?retryWrites=true&w=majority&appName=WeHack-Cluster"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["WeHack"]
collection = db["user-A-properties"]

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
    .main {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Your Co-Venturer Matches")

# Fetch and sort User A entries by risk_score (ascending = lowest first)
results = list(collection.find().sort("risk_score", 1))

if results:
    for result in results:
        name = result.get("first_name", "Unknown")
        email = result.get("email", "N/A")
        lat = result.get("lat", "N/A")
        lon = result.get("lon", "N/A")
        risk_score = result.get("risk_score", "N/A")

        # Display each match in an expandable box
        with st.expander(f"{name}"):
            st.write(f"📍 **Coordinates:** {lat}, {lon}")
            st.write(f"📧 **Email:** `{email}`")
            st.write(f"📊 **Risk Score:** `{risk_score:.2f}`" if isinstance(risk_score, (int, float)) else f"📊 Risk Score: {risk_score}")
            st.write("📝 **More Info:** You can add a short property/business description here.")
else:
    st.info("No matches found in the database")'''



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
    "Address": ["6580 Lake Worth Blvd, Lake Worth, TX", "4730 SW Loop 820, Fort Worth, TX", "6000 Quebec Street, Fort Worth, TX", "1230 River Bend Drive, Dallas, TX", "1503 LBJ Fwy, Farmers Branch, TX"],
    "Email": ["john.doe@email.com", "jane.smith@email.com", "mark.johnson@email.com", "emily.davis@email.com", "michael.brown@email.com"],
    "Contact": ["123-456-7890", "234-567-8901", "345-678-9012", "456-789-0123", "567-890-1234"],
    "Location": ["Dallas, TX", "Fort Worth, TX", "Arlington, TX", "Plano, TX", "Irving, TX"],
    "Risk Score": ["0.23", "0.33", "0.34", "0.54", "0.66"]

}

# Create a DataFrame
df = pd.DataFrame(data)

# Display each user in a clickable box
for idx, row in df.iterrows():
    with st.expander(f"{row['Name']}"):
        st.write(f"**Address:** {row['Address']}")
        st.write(f"**Email:** {row['Email']}")
        st.write(f"**Contact:** {row['Contact']}")
        st.write(f"**Location:** {row['Location']}")
        st.write(f"**Risk Score:** {row['Risk Score']}")
        st.write(f"**More Information:** Here you can add more details about the user.")


