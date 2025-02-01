import streamlit as st
from models.api import get_response
import json

def show_guide():
    st.header("Passport Guidance")
    st.write("Follow these steps to obtain or update your Passport.")

    with open("data/process.json", "r") as f:
        steps = json.load(f)
    passport_steps = steps["Passport"]

    for i, step in enumerate(passport_steps, 1):
        st.markdown(f"{i}. **{step}**")

    st.subheader("Frequently Asked Questions (FAQs)")
    with open("data/faq.json", "r") as f:
        faqs = json.load(f)
    passport_faqs = faqs["Passport"]

    for faq in passport_faqs:
        st.markdown(f"**Q: {faq['question']}**")
        st.write(f"A: {faq['answer']}")

    st.info("For personalized assistance, type your query below:")
    query = st.text_input("Enter your query")
    if query:
        response = get_response(query)
        st.write(response)