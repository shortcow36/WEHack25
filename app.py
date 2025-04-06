import streamlit as st

st.set_page_config(
    page_title="Landing Page",
    page_icon="👋",
)

st.title("Welcome to Our Awesome App!")

if st.button("Sign In"):
    st.write("You clicked the Sign In button!")

if st.button("Register"):
    st.write("You clicked the Register button!")