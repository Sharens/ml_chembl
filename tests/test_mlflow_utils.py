from __future__ import annotations

from src.mlflow_utils import (
    _normalize_tracking_uri,
    configure_mlflow,
    log_epoch_metrics,
    log_final_metrics,
    log_split_sizes,
    start_training_run,
)


class TestNormalizeTrackingUri:
    def test_relative_sqlite(self):
        result = _normalize_tracking_uri("sqlite:///mlflow.db")
        assert result.startswith("sqlite:////")
        assert result.endswith("mlflow.db")

    def test_absolute_sqlite(self):
        result = _normalize_tracking_uri("sqlite:////abs/path/mlflow.db")
        assert result == "sqlite:////abs/path/mlflow.db"

    def test_non_sqlite_passthrough(self):
        result = _normalize_tracking_uri("http://localhost:5000")
        assert result == "http://localhost:5000"

    def test_empty_string(self):
        assert _normalize_tracking_uri("") == ""


class TestConfigureMlflow:
    def test_creates_experiment_when_missing(self, mocker, tmp_path):
        mock_set_uri = mocker.patch("src.mlflow_utils.mlflow.set_tracking_uri")
        mock_set_exp = mocker.patch("src.mlflow_utils.mlflow.set_experiment")
        mock_get = mocker.patch(
            "src.mlflow_utils.mlflow.get_experiment_by_name", return_value=None
        )
        mock_create = mocker.patch(
            "src.mlflow_utils.mlflow.create_experiment", return_value="new_id"
        )

        exp_id = configure_mlflow(
            tracking_uri="sqlite:///test.db",
            experiment_name="test_exp",
            artifact_root=str(tmp_path / "mlruns"),
        )

        assert exp_id == "new_id"
        mock_set_uri.assert_called_once()
        mock_get.assert_called_once_with("test_exp")
        mock_create.assert_called_once()
        mock_set_exp.assert_called_once_with("test_exp")

    def test_uses_existing_experiment(self, mocker, tmp_path):
        mocker.patch(
            "src.mlflow_utils.mlflow.get_experiment_by_name",
            return_value=mocker.Mock(experiment_id="existing_id"),
        )
        mock_create = mocker.patch("src.mlflow_utils.mlflow.create_experiment")

        exp_id = configure_mlflow(
            experiment_name="existing_exp",
            artifact_root=str(tmp_path / "mlruns"),
        )

        assert exp_id == "existing_id"
        mock_create.assert_not_called()


class TestStartTrainingRun:
    def test_sets_tags(self, mocker):
        mock_run = mocker.patch("src.mlflow_utils.mlflow.start_run")
        mock_set_tags = mocker.patch("src.mlflow_utils.mlflow.set_tags")

        with start_training_run(
            run_name="test_run",
            model_type="GNN",
            split_type="scaffold",
            seed=42,
        ):
            pass

        mock_run.assert_called_once_with(run_name="test_run")
        mock_set_tags.assert_called_once_with(
            {
                "project": "ml_chembl",
                "task": "bioactivity_regression",
                "target": "pIC50",
                "model_type": "GNN",
                "split_type": "scaffold",
                "seed": "42",
            }
        )

    def test_sets_tags_with_extra(self, mocker):
        mock_set_tags = mocker.patch("src.mlflow_utils.mlflow.set_tags")

        with start_training_run(
            run_name="test",
            model_type="MLP",
            split_type="random",
            seed=7,
            extra_tags={"dataset": "chembl"},
        ):
            pass

        called_tags = mock_set_tags.call_args[0][0]
        assert called_tags["dataset"] == "chembl"
        assert called_tags["model_type"] == "MLP"


class TestLogSplitSizes:
    def test_logs_metrics(self, mocker):
        mock_log_metrics = mocker.patch("src.mlflow_utils.mlflow.log_metrics")

        log_split_sizes(train_size=100, val_size=20, test_size=30)

        mock_log_metrics.assert_called_once_with(
            {"n_train": 100.0, "n_val": 20.0, "n_test": 30.0}
        )


class TestLogEpochMetrics:
    def test_logs_all_keys(self, mocker):
        mock_log_metric = mocker.patch("src.mlflow_utils.mlflow.log_metric")

        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_r2": 0.8, "lr": 0.01},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "val_r2": 0.85},
        ]

        log_epoch_metrics(history)

        assert mock_log_metric.call_count == 7

    def test_handles_missing_keys(self, mocker):
        mock_log_metric = mocker.patch("src.mlflow_utils.mlflow.log_metric")

        log_epoch_metrics([{"epoch": 1, "train_loss": 0.5}])

        mock_log_metric.assert_called_once_with("train_loss", 0.5, step=1)

    def test_empty_history(self, mocker):
        mock_log_metric = mocker.patch("src.mlflow_utils.mlflow.log_metric")

        log_epoch_metrics([])

        mock_log_metric.assert_not_called()


class TestLogFinalMetrics:
    def test_without_r2_test(self, mocker):
        mock_log = mocker.patch("src.mlflow_utils.mlflow.log_metric")

        log_final_metrics(best_val_loss=0.5, r2_val=0.85)

        assert mock_log.call_count == 2
        mock_log.assert_any_call("best_val_loss", 0.5)
        mock_log.assert_any_call("r2_val", 0.85)

    def test_with_r2_test(self, mocker):
        mock_log = mocker.patch("src.mlflow_utils.mlflow.log_metric")

        log_final_metrics(best_val_loss=0.5, r2_val=0.85, r2_test=0.82)

        assert mock_log.call_count == 3
        mock_log.assert_any_call("r2_test", 0.82)
