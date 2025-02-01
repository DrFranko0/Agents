import streamlit as st
from models.api import get_response
import json

def show_guide():
    st.header("DL Guidance")
    st.write("Follow these steps to obtain or update your DL.")

    with open("data/process.json", "r") as f:
        steps = json.load(f)
    dl_steps = steps["DL"]

    for i, step in enumerate(dl_steps, 1):
        st.markdown(f"{i}. **{step}**")

    st.subheader("Frequently Asked Questions (FAQs)")
    with open("data/faq.json", "r") as f:
        faqs = json.load(f)
    dl_faqs = faqs["DL"]

    for faq in dl_faqs:
        st.markdown(f"**Q: {faq['question']}**")
        st.write(f"A: {faq['answer']}")

    st.info("For personalized assistance, type your query below:")
    query = st.text_input("Enter your query")
    if query:
        response = get_response(query)
        st.write(response)
