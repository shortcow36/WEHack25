
import streamlit as st
from Linear_Regression import predict_risk_score  # adjust path if needed
import plotly.graph_objects as go



st.set_page_config(page_title="Risk Score", page_icon="📈")
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
    /* Optional: Style the main content area */
    .main {
        color: white;  /* Change text color for contrast */
    }
    </style>
""", unsafe_allow_html=True)



st.title("Risk Score:")

st.write("This is where the risk analysis will be displayed.")


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


# Example risk score (to be replaced with dynamic calculation)
risk_score = 74  # This is a sample risk score, replace it with the actual calculation

# Determine risk level based on the score
if risk_score >= 65:
    risk_level = "High Risk"
elif risk_score >= 25:
    risk_level = "Medium Risk"
else:
    risk_level = "Low Risk"

# Create a gauge chart with more polished visual styling
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={'text': f" {risk_level}", 'font': {'size': 24, 'color': "white", 'weight' : 100}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white"},
        'bar': {'color': "#ffffff", 'thickness': 0.3},  # Color for the gauge bar
        'bgcolor': "#111827",  # Background color of the gauge
        'borderwidth': 2,
        'bordercolor': "white",
        'steps': [
            {'range': [0, 25], 'color': "#28a745"},  # Green for low risk
            {'range': [25, 65], 'color': "#ffc300"},  # Yellow for medium risk
            {'range': [65, 100], 'color': "#ff5733"}  # Red for high risk
        ]
    }
))

# Set layout for the overall figure (background color, font style, and height)
fig.update_layout(
    paper_bgcolor="black",  # Background color for the entire figure
    font={'color': "white", 'family': "Arial, sans-serif"},  # Font settings
    height=300,  # Adjust the height of the chart
)

# Display the gauge chart in Streamlit
st.plotly_chart(fig)

