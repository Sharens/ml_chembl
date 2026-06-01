from __future__ import annotations

from agent.model_inference import load_model, predict_pic50

MODEL_NAME = "gemma4:e4b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_pIC50",
            "description": "Predict the bioactivity pIC50 value for a molecule given its SMILES string. Returns the predicted pIC50 (higher = more active).",
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
    }
]

SYSTEM_PROMPT = (
    "You are a chemistry assistant specializing in predicting bioactivity of molecules. "
    "Your primary tool is a Graph Neural Network (GIN) model trained on ChEMBL data. "
    "When a user provides a SMILES string, use the predict_pIC50 tool to get the prediction. "
    "Interpret the pIC50 value in context: pIC50 = -log10(IC50), so higher values mean higher potency. "
    "pIC50 < 5 means low potency (mM range), 5-7 means moderate (μM range), >7 means high potency (nM range). "
    "If the SMILES is invalid, explain what went wrong."
)

_gnn_model = None


def _ensure_model():
    global _gnn_model
    if _gnn_model is None:
        loaded = load_model()
        if loaded is not None:
            _gnn_model, _ = loaded
    return _gnn_model


def _execute_tool_call(tool_call: dict) -> str:
    fn_name = tool_call.get("function", {}).get("name", "")
    args = tool_call.get("function", {}).get("arguments", {})

    if fn_name == "predict_pIC50":
        smiles = args.get("smiles", "")
        model = _ensure_model()
        result = predict_pic50(smiles, model=model)
        if result["valid"]:
            return (
                f"The predicted pIC50 for molecule {result['smiles']} is {result['pIC50']}. "
                f"This value is -log10(IC50), so higher = more potent."
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
