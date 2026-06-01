from __future__ import annotations

from src.agent.llm_agent import _ensure_model, _execute_tool_call, run_agent


class TestEnsureModel:
    def test_returns_none_when_model_fails(self, mocker):
        mocker.patch("src.agent.llm_agent.load_model", return_value=None)
        mocker.patch("src.agent.llm_agent._gnn_model", None)

        result = _ensure_model()
        assert result is None

    def test_loads_model_when_none(self, mocker):
        mock_model = mocker.Mock()
        mocker.patch(
            "src.agent.llm_agent.load_model",
            return_value=(mock_model, {"r2_val": 0.8}),
        )
        mocker.patch("src.agent.llm_agent._gnn_model", None)

        result = _ensure_model()
        assert result is mock_model

    def test_uses_cached_model(self, mocker):
        mock_model = mocker.Mock()
        mocker.patch("src.agent.llm_agent.load_model")
        mocker.patch("src.agent.llm_agent._gnn_model", mock_model)

        result = _ensure_model()
        assert result is mock_model


class TestExecuteToolCall:
    def test_predict_pIC50_valid(self, mocker):
        mocker.patch(
            "src.agent.llm_agent.predict_pic50",
            return_value={
                "smiles": "CCO",
                "pIC50": 4.5,
                "valid": True,
                "error": None,
            },
        )

        tool_call = {
            "function": {
                "name": "predict_pIC50",
                "arguments": {"smiles": "CCO"},
            }
        }
        result = _execute_tool_call(tool_call)
        assert "4.5" in result
        assert "CCO" in result

    def test_predict_pIC50_invalid(self, mocker):
        mocker.patch(
            "src.agent.llm_agent.predict_pic50",
            return_value={
                "smiles": "INVALID",
                "pIC50": None,
                "valid": False,
                "error": "Invalid SMILES string",
            },
        )

        tool_call = {
            "function": {
                "name": "predict_pIC50",
                "arguments": {"smiles": "INVALID"},
            }
        }
        result = _execute_tool_call(tool_call)
        assert "Error" in result

    def test_unknown_tool(self):
        tool_call = {
            "function": {
                "name": "unknown_tool",
                "arguments": {},
            }
        }
        result = _execute_tool_call(tool_call)
        assert "Unknown tool" in result


class TestRunAgent:
    def test_ollama_not_installed(self, mocker):
        mocker.patch("src.agent.llm_agent._ensure_model")
        mocker.patch.dict("sys.modules", {"ollama": None})
        import src.agent.llm_agent as llm

        if "ollama" in llm.__dict__:
            mocker.patch.object(llm, "ollama", None)

        result = run_agent("test")
        assert "ollama" in result.lower()

    def test_no_tool_calls(self, mocker):
        mocker.patch("src.agent.llm_agent._ensure_model")
        mock_chat = mocker.patch("ollama.chat")
        mock_chat.return_value = {"message": {"content": "Direct response"}}

        result = run_agent("test")
        assert result == "Direct response"

    def test_with_tool_calls(self, mocker):
        mocker.patch("src.agent.llm_agent._ensure_model")
        mock_chat = mocker.patch("ollama.chat")
        mock_chat.side_effect = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "predict_pIC50",
                                "arguments": {"smiles": "CCO"},
                            }
                        }
                    ],
                }
            },
            {"message": {"content": "Final answer"}},
        ]

        result = run_agent("test")
        assert result == "Final answer"
        assert mock_chat.call_count == 2
