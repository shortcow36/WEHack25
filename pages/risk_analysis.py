import streamlit as st
import plotly.graph_objects as go

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