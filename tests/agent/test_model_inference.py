from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.agent.model_inference import (
    GNNRegressor,
    _get_best_model_path,
    _infer_architecture,
    load_model,
    mol_to_graph,
    one_hot_encode,
    predict_pic50,
)


class TestOneHotEncode:
    def test_matches_category(self):
        assert one_hot_encode(2, [1, 2, 3]) == [0, 1, 0]

    def test_no_match(self):
        assert one_hot_encode(5, [1, 2, 3]) == [0, 0, 0]

    def test_empty_categories(self):
        assert one_hot_encode(1, []) == []


class TestMolToGraph:
    def test_valid_smiles(self):
        data = mol_to_graph("CCO")
        assert data is not None
        assert data.x.shape[0] == 3
        assert data.edge_index.shape[1] > 0

    def test_invalid_smiles(self):
        data = mol_to_graph("not_valid")
        assert data is None

    def test_empty_string(self):
        data = mol_to_graph("")
        assert data is None or data.x.shape[0] == 0

    def test_aromatic_ring(self):
        data = mol_to_graph("c1ccccc1")
        assert data is not None
        assert data.x.shape[0] == 6

    def test_no_bonds_single_atom(self):
        data = mol_to_graph("[Ar]")
        if data is not None:
            assert data.x.shape[0] == 1


class TestGNNRegressor:
    def test_init_default_pooling(self):
        model = GNNRegressor(50, 11)
        assert model.pooling == "mean"
        assert len(model.convs) == 4

    def test_init_add_pooling(self):
        model = GNNRegressor(50, 11, pooling="add")
        assert model.pooling == "add"

    def test_init_invalid_pooling(self):
        with pytest.raises(ValueError, match="pooling"):
            GNNRegressor(50, 11, pooling="max")

    def test_forward_shape(self):
        model = GNNRegressor(10, 4, hidden_dim=16, num_layers=2)
        model.eval()
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 4)
        batch = torch.zeros(5, dtype=torch.long)
        from torch_geometric.data import Data

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        out = model(data)
        assert out.shape == (1, 1)


class TestInferArchitecture:
    def test_basic(self):
        model = GNNRegressor(50, 11, hidden_dim=128, num_layers=4)
        state_dict = model.state_dict()
        arch = _infer_architecture(state_dict)
        assert arch["node_features"] == 50
        assert arch["edge_features"] == 11
        assert arch["hidden_dim"] == 128
        assert arch["num_layers"] == 4

    def test_different_dimensions(self):
        model = GNNRegressor(30, 6, hidden_dim=64, num_layers=2)
        arch = _infer_architecture(model.state_dict())
        assert arch["node_features"] == 30
        assert arch["edge_features"] == 6
        assert arch["hidden_dim"] == 64
        assert arch["num_layers"] == 2


class TestGetBestModelPath:
    def test_no_cache_dir(self, mocker):
        mocker.patch(
            "src.agent.model_inference.MODEL_CACHE_DIR",
            Path("/nonexistent"),
        )
        result = _get_best_model_path()
        assert result is None

    def test_picks_best_r2(self, mocker, tmp_model_dir):
        import torch

        torch.save(
            {"result": {"model": "GNN", "split": "scaffold", "r2_val": 0.5}},
            tmp_model_dir / "default_a.pt",
        )
        torch.save(
            {"result": {"model": "GNN", "split": "scaffold", "r2_val": 0.9}},
            tmp_model_dir / "default_b.pt",
        )
        mocker.patch("src.agent.model_inference.MODEL_CACHE_DIR", tmp_model_dir)

        best = _get_best_model_path()
        assert best is not None
        assert "default_b" in best.name

    def test_skips_corrupted(self, mocker, tmp_model_dir):
        (tmp_model_dir / "corrupted.pt").write_text("not a torch file")
        torch.save(
            {"result": {"model": "GNN", "split": "scaffold", "r2_val": 0.7}},
            tmp_model_dir / "default_good.pt",
        )
        mocker.patch("src.agent.model_inference.MODEL_CACHE_DIR", tmp_model_dir)

        best = _get_best_model_path()
        assert best is not None
        assert "default_good" in best.name


class TestLoadModel:
    def test_returns_none_when_no_model(self, mocker):
        mocker.patch(
            "src.agent.model_inference._get_best_model_path",
            return_value=None,
        )
        result = load_model()
        assert result is None

    def test_loads_model_successfully(self, mocker, tmp_model_dir):
        import torch

        dummy_model = GNNRegressor(50, 11, hidden_dim=32, num_layers=2)
        ckpt = {
            "model_state_dict": dummy_model.state_dict(),
            "result": {
                "model": "GNN",
                "split": "scaffold",
                "pooling": "mean",
                "gnn_dropout": 0.15,
                "r2_val": 0.8,
            },
        }
        model_path = tmp_model_dir / "model.pt"
        torch.save(ckpt, model_path)

        model, result = load_model(model_path=model_path)
        assert model is not None
        assert result["r2_val"] == 0.8


class TestPredictPIC50:
    def test_invalid_smiles(self):
        result = predict_pic50("not_valid", model=None)
        assert result["valid"] is False
        assert "Invalid SMILES" in result["error"]

    def test_with_model_cpu(self, mocker):
        mocker.patch(
            "src.agent.model_inference.torch.device",
            return_value=torch.device("cpu"),
        )
        model = GNNRegressor(28, 11, hidden_dim=32, num_layers=2)
        result = predict_pic50("CCO", model=model)
        assert result["valid"] is True
        assert result["pIC50"] is not None
        assert isinstance(result["pIC50"], float)

    def test_with_mocked_model(self, mocker):
        mock_model = mocker.Mock()
        mock_model.return_value = mocker.Mock()
        mock_model.return_value.cpu.return_value.item.return_value = 5.5
        mock_data = mocker.Mock()
        mocker.patch(
            "src.agent.model_inference.mol_to_graph",
            return_value=mock_data,
        )

        result = predict_pic50("CCO", model=mock_model)
        assert result["valid"] is True
        assert result["pIC50"] == 5.5
