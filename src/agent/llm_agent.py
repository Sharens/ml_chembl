from __future__ import annotations

from functools import cache

from src.agent.model_inference import load_model, predict_pic50
from src.agent.rdkit_tools import (
    compute_descriptors,
    compute_logp,
    compute_molecular_weight,
)

MODEL_NAME = "gemma4:e4b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_pIC50",
            "description": (
                "Predict the bioactivity pIC50 value for a molecule given its SMILES string. "
                "Returns the predicted pIC50 (higher = more active). "
                "Use this to evaluate biological potency of the compound."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES representation of the molecule",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_molecular_weight",
            "description": (
                "Compute the molecular weight (g/mol) of a molecule from its SMILES string "
                "using RDKit. Useful for drug-likeness assessment (Lipinski rule: MW < 500)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES representation of the molecule",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_logp",
            "description": (
                "Compute the octanol-water partition coefficient (LogP) of a molecule "
                "from its SMILES string using RDKit. Useful for drug-likeness assessment "
                "(Lipinski rule: LogP < 5). Higher LogP = more lipophilic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES representation of the molecule",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_descriptors",
            "description": (
                "Compute a comprehensive set of molecular descriptors for a molecule "
                "from its SMILES string using RDKit. Returns: molecular weight, LogP, "
                "H-bond acceptors, H-bond donors, polar surface area (PSA), "
                "rotatable bonds, heavy atom count, and aromatic ring count. "
                "Use this for full physicochemical profiling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES representation of the molecule",
                    }
                },
                "required": ["smiles"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a chemistry assistant specializing in predicting bioactivity and computing "
    "molecular properties of drug-like molecules. You have access to these tools:\n"
    "1. predict_pIC50 — uses a Graph Neural Network (GIN) trained on ChEMBL data to predict bioactivity\n"
    "2. compute_molecular_weight — computes molecular weight via RDKit\n"
    "3. compute_logp — computes LogP (lipophilicity) via RDKit\n"
    "4. compute_descriptors — computes full physicochemical profile via RDKit\n\n"
    "When a user provides a SMILES string, you should:\n"
    "- Use predict_pIC50 to get the bioactivity prediction\n"
    "- Use compute_molecular_weight and compute_logp to assess drug-likeness\n"
    "- Interpret pIC50 in context: pIC50 = -log10(IC50), higher = more potent\n"
    "  pIC50 < 5 = low potency (mM), 5-7 = moderate (μM), >7 = high potency (nM)\n"
    "- Apply Lipinski's Rule of 5: MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10\n"
    "- Combine prediction results with molecular properties to give a complete assessment\n"
    "- If the SMILES is invalid, explain what went wrong\n"
    "Always provide a comprehensive answer that integrates bioactivity prediction "
    "with molecular properties."
)


@cache
def _get_model():
    loaded = load_model()
    if loaded is not None:
        return loaded[0]
    return None


def _ensure_model():
    return _get_model()


def _clear_model_cache():
    _get_model.cache_clear()


def _execute_tool_call(tool_call: dict) -> str:
    fn_name = tool_call.get("function", {}).get("name", "")
    args = tool_call.get("function", {}).get("arguments", {})

    if fn_name == "predict_pIC50":
        smiles = args.get("smiles", "")
        model = _ensure_model()
        result = predict_pic50(smiles, model=model)
        if result["valid"]:
            pIC50 = result["pIC50"]
            potency = (
                "low (mM range)"
                if pIC50 < 5
                else "moderate (μM range)"
                if pIC50 < 7
                else "high (nM range)"
            )
            return (
                f"Predicted pIC50 for {result['smiles']}: {pIC50} ({potency}). "
                f"pIC50 = -log10(IC50); higher means more potent."
            )
        else:
            return f"Error: {result['error']}"

    elif fn_name == "compute_molecular_weight":
        smiles = args.get("smiles", "")
        result = compute_molecular_weight(smiles)
        if result["valid"]:
            mw = result["molecular_weight"]
            ro5 = "passes" if mw <= 500 else "violates"
            return f"Molecular weight of {result['smiles']}: {mw} g/mol (Lipinski {ro5} MW < 500 rule)."
        else:
            return f"Error: {result['error']}"

    elif fn_name == "compute_logp":
        smiles = args.get("smiles", "")
        result = compute_logp(smiles)
        if result["valid"]:
            logp = result["logp"]
            ro5 = "passes" if logp <= 5 else "violates"
            return f"LogP of {result['smiles']}: {logp} (Lipinski {ro5} LogP < 5 rule). Higher = more lipophilic."
        else:
            return f"Error: {result['error']}"

    elif fn_name == "compute_descriptors":
        smiles = args.get("smiles", "")
        result = compute_descriptors(smiles)
        if result["valid"]:
            return (
                f"Physicochemical profile for {result['smiles']}:\n"
                f"  Molecular weight: {result['molecular_weight']} g/mol\n"
                f"  LogP: {result['logp']}\n"
                f"  H-bond acceptors: {result['hba']}\n"
                f"  H-bond donors: {result['hbd']}\n"
                f"  Polar surface area: {result['psa']} Å²\n"
                f"  Rotatable bonds: {result['rotatable_bonds']}\n"
                f"  Heavy atoms: {result['heavy_atoms']}\n"
                f"  Aromatic rings: {result['aromatic_rings']}"
            )
        else:
            return f"Error: {result['error']}"

    return f"Unknown tool: {fn_name}"


def run_agent(user_input: str) -> str:
    try:
        import ollama
    except ImportError:
        return "Error: ollama Python package is not installed. Run: uv add ollama"

    _ensure_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response = ollama.chat(model=MODEL_NAME, messages=messages, tools=TOOLS)

    message = response["message"]

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            tool_result = _execute_tool_call(tc)
            messages.append(message)
            messages.append({"role": "tool", "content": tool_result})

        final = ollama.chat(model=MODEL_NAME, messages=messages)
        return final["message"]["content"]

    return message["content"]
