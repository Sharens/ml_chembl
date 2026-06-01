from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def mock_mlflow(mocker):
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.set_experiment")
    mocker.patch("mlflow.get_experiment_by_name", return_value=None)
    mocker.patch("mlflow.create_experiment", return_value="exp_123")
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.set_tags")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_metrics")
    return mocker


@pytest.fixture
def sample_polars_df():
    return pl.DataFrame(
        {
            "canonical_smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
            "standard_value": [100.0, 50.0, None],
            "standard_units": ["nM", None, "nM"],
            "pchembl_value": [None, None, 5.5],
            "standard_type": ["IC50", "IC50", "IC50"],
        }
    )


@pytest.fixture
def mock_torch_state_dict():
    import torch.nn as nn

    model = nn.Linear(10, 5)
    return model.state_dict()


@pytest.fixture
def tmp_model_dir(tmp_path):
    d = tmp_path / "model_cache"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def mock_httpx_client(mocker):
    mock_response = mocker.MagicMock()
    mock_response.headers = {"content-length": "1000"}
    mock_response.iter_bytes.return_value = [b"x" * 500, b"y" * 500]
    mock_stream = mocker.MagicMock()
    mock_stream.__enter__.return_value = mock_response
    mock_client_instance = mocker.MagicMock()
    mock_client_instance.stream.return_value = mock_stream
    mock_client_class = mocker.patch("httpx.Client", return_value=mock_client_instance)
    return mock_client_class
