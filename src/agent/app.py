from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.agent.llm_agent import run_agent

st.set_page_config(
    page_title="Bioactivity Predictor",
    page_icon="🧪",
    layout="centered",
)

st.markdown(
    """
    <style>
        #stStatusWidget, [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 Bioactivity Predictor (pIC50)")
st.markdown(
    "Enter a SMILES string to predict its bioactivity using a **GIN neural network** "
    "trained on ChEMBL data, with **LLM-powered interpretation**."
)

smiles = st.text_input(
    "SMILES",
    value="CC(=O)OC1=CC=CC=C1C(=O)O",
    help="SMILES notation of the molecule",
)

predict = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict and smiles:
    status_placeholder = st.empty()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check the input.")
        st.stop()

    img = Draw.MolToImage(mol, size=(400, 300))
    st.image(img, caption="Molecule 2D Structure", use_container_width=False)

    with status_placeholder.container():
        st.caption("LLM agent is processing...")

    response = run_agent(smiles)
    status_placeholder.empty()

    st.subheader("Agent Response")
    st.markdown(response)
elif predict and not smiles:
    st.warning("Please enter a SMILES string.")

st.divider()
st.markdown(
    """
    **How it works:**
    1. You enter a SMILES string
    2. An **LLM agent** (Gemma4) receives it and decides to call the prediction tool
    3. The **GIN neural network** (trained on ChEMBL, scaffold split) predicts pIC50
    4. The LLM interprets the result in chemical context

    *Powered by GIN + Ollama + Streamlit*
    """
)
