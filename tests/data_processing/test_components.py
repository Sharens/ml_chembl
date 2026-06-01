from __future__ import annotations

import math

import polars as pl

from src.data_processing.components import (
    Config,
    DataDownloader,
    DataLoader,
    DataMisc,
    DataProcessor,
)


class TestConfig:
    def test_default_values(self):
        cfg = Config()
        assert cfg.archive_name == "chembl_36_sqlite.tar.gz"
        assert "chembl" in cfg.download_url

    def test_custom_data_path(self, tmp_path):
        cfg = Config(data_path=tmp_path)
        assert cfg.data_path == tmp_path


class TestDataMisc:
    def test_compute_pIC50_with_nM(self, sample_polars_df):
        result = DataMisc.compute_pIC50(sample_polars_df)
        pic50 = (
            result.filter(pl.col("canonical_smiles") == "CCO").select("pIC50").item()
        )
        expected = -math.log10(100.0 * 1e-9)
        assert abs(pic50 - expected) < 1e-6

    def test_compute_pIC50_preserves_existing(self, sample_polars_df):
        result = DataMisc.compute_pIC50(sample_polars_df)
        pic50 = (
            result.filter(pl.col("canonical_smiles") == "c1ccccc1")
            .select("pIC50")
            .item()
        )
        assert pic50 == 5.5

    def test_compute_pIC50_null_standard_value(self, sample_polars_df):
        result = DataMisc.compute_pIC50(sample_polars_df)
        pic50 = (
            result.filter(pl.col("canonical_smiles") == "c1ccccc1")
            .select("pIC50")
            .item()
        )
        assert pic50 == 5.5

    def test_impute_units_missing_in_range(self, sample_polars_df):
        result = DataMisc.impute_units(sample_polars_df)
        val = (
            result.filter(pl.col("canonical_smiles") == "CC(=O)O")
            .select("standard_units")
            .item()
        )
        assert val == "nM"

    def test_impute_units_preserves_existing(self, sample_polars_df):
        result = DataMisc.impute_units(sample_polars_df)
        val = (
            result.filter(pl.col("canonical_smiles") == "CCO")
            .select("standard_units")
            .item()
        )
        assert val == "nM"

    def test_impute_units_out_of_range(self):
        df = pl.DataFrame(
            {
                "standard_value": [1e9],
                "standard_units": [None],
            }
        )
        result = DataMisc.impute_units(df)
        assert result["standard_units"].item() is None


class TestDataDownloader:
    def test_output_path(self, tmp_path):
        cfg = Config(data_path=tmp_path)
        d = DataDownloader(cfg)
        assert d.output_path == tmp_path / "chembl_36.db"

    def test_download_already_exists(self, mocker, tmp_path):
        cfg = Config(data_path=tmp_path, archive_name="existing.tar.gz")
        d = DataDownloader(cfg)
        mocker.patch("os.path.exists", return_value=True)
        mock_log = mocker.patch("logging.info")

        d.download_sqlite_archive()

        mock_log.assert_called_once_with("Archiwum już znajduje się na dysku.")

    def test_download_success(self, mocker, tmp_path, mock_httpx_client):
        cfg = Config(data_path=tmp_path, archive_name="test.tar.gz")
        d = DataDownloader(cfg)
        mocker.patch("os.path.exists", return_value=False)
        mock_open = mocker.mock_open()
        mocker.patch("builtins.open", mock_open)

        d.download_sqlite_archive()

        assert mock_open.called

    def test_cleanup_removes_archive(self, mocker, tmp_path):
        cfg = Config(data_path=tmp_path, archive_name="test.tar.gz")
        d = DataDownloader(cfg)
        mock_remove = mocker.patch("os.remove")

        archive_path = tmp_path / "test.tar.gz"
        archive_path.touch()
        mocker.patch("os.path.exists", return_value=True)

        d._cleanup()

        mock_remove.assert_called_once_with(cfg.archive_name)

    def test_cleanup_skips_when_missing(self, mocker, tmp_path):
        cfg = Config(data_path=tmp_path)
        d = DataDownloader(cfg)
        mock_remove = mocker.patch("os.remove")
        mocker.patch("os.path.exists", return_value=False)

        d._cleanup()

        mock_remove.assert_not_called()


class TestDataLoader:
    def test_load_from_sqlite(self, mocker, tmp_path):
        fake_df = pl.DataFrame({"activity_id": [1, 2]})
        mock_read = mocker.patch(
            "src.data_processing.components.pl.read_database_uri",
            return_value=fake_df,
        )

        cfg = Config(data_path=tmp_path)
        loader = DataLoader(cfg)
        result = loader.load_from_sqlite()

        assert result.height == 2
        mock_read.assert_called_once()
        assert "SELECT" in mock_read.call_args[1]["query"]


class TestDataProcessor:
    def test_process_data_no_sampling(self, sample_polars_df):
        cfg = Config()
        proc = DataProcessor(cfg)
        result = proc.process_data(sample_polars_df)
        assert result.height <= sample_polars_df.height
        assert "pIC50" in result.columns

    def test_process_data_with_sampling(self, sample_polars_df):
        cfg = Config()
        proc = DataProcessor(cfg)
        result = proc.process_data(sample_polars_df, n_value=2)
        assert result.height <= 2
        assert "pIC50" in result.columns
