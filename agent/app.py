from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from agent.llm_agent import run_agent

st.set_page_config(
    page_title="Bioactivity Predictor",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 Bioactivity Predictor (pIC50)")
st.markdown(
    "Enter a SMILES string to predict its bioactivity using a **GIN neural network** "
    "trained on ChEMBL data, with **LLM-powered interpretation**."
)

smiles = st.text_input(
    "SMILES",
    placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
    help="SMILES notation of the molecule",
)

col1, col2 = st.columns([1, 5])
with col1:
    predict = st.button("🔮 Predict", type="primary", use_container_width=True)
with col2:
    examples = st.selectbox(
        "Or try an example:",
        [
            "",
            "Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Paracetamol: CC(=O)NC1=CC=C(C=C1)O",
        ],
    )

if examples:
    smiles = examples.split(": ", 1)[1]

if predict and smiles:
    with st.spinner("Running GNN model and LLM interpretation..."):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string. Please check the input.")
            st.stop()

        img = Draw.MolToImage(mol, size=(400, 300))
        st.image(img, caption="Molecule 2D Structure", use_container_width=False)

        with st.status("Agent at work...", expanded=True) as status:
            st.write("🧠 LLM agent analyzing SMILES...")
            st.write("⚙️ Calling GNN model via tool...")
            response = run_agent(smiles)
            status.update(label="Done!", state="complete", expanded=True)

        st.subheader("Agent Response")
        st.markdown(response)
elif predict and not smiles:
    st.warning("Please enter a SMILES string.")

st.divider()
st.markdown(
    """
    **How it works:**
    1. You enter a SMILES string
    2. An **LLM agent** (Qwen2.5) receives it and decides to call the prediction tool
    3. The **GIN neural network** (trained on ChEMBL, scaffold split) predicts pIC50
    4. The LLM interprets the result in chemical context

    *Powered by GIN + Ollama + Streamlit*
    """
)
