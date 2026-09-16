"""Minimal chat UI. Run the API first: `make api`, then `make ui`."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Norvik After-Sales Assistant", page_icon="🔧")
st.title("Norvik after-sales assistant")
st.caption("Answers come only from the service manuals, warranty policy and recall database, with sources.")

with st.sidebar:
    model = st.selectbox("Vehicle", ["Not specified", "Aster Hybrid", "Tern EV"])
    year = st.number_input("Model year (optional)", min_value=0, max_value=2030, value=0, step=1)

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

if question := st.chat_input("e.g. The charge port light is flashing red"):
    st.chat_message("user").markdown(question)
    payload = {
        "question": question,
        "model": None if model == "Not specified" else model,
        "year": year or None,
    }
    try:
        data = requests.post(f"{API_URL}/ask", json=payload, timeout=60).json()
        reply = data["answer"]
        if data["citations"]:
            reply += "\n\n**Sources**\n" + "\n".join(f"- {c}" for c in data["citations"])
        reply += f"\n\n`route: {data['route']}`"
    except requests.RequestException as exc:
        reply = f"Could not reach the API at {API_URL}: {exc}"
    st.chat_message("assistant").markdown(reply)
    st.session_state.history += [("user", question), ("assistant", reply)]
