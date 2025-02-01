import streamlit as st
from components import adhaar, pan, residence, drivinglicence, passport

st.set_page_config(page_title="PoliAi", layout="wide")

st.title("PoliAi: Simplifying Bureaucracy for Students in India")

options = ["Home", "Aadhaar", "PAN", "Residence Permit", "Passport", "Driving Licence"]
choice = st.sidebar.selectbox("Select a Service", options)

if choice == "Home":
    st.header("Welcome to PoliAi")
    st.write("Choose a service from the sidebar to get started.")

elif choice == "Aadhaar":
    adhaar.show_guide()

elif choice == "PAN":
    pan.show_guide()

elif choice == "Driving Licence":
    drivinglicence.show_guide()

elif choice == "PAN Guidance":
    passport.show_guide()

elif choice == "Passport":
    residence.show_guide()
