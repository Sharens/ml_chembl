from __future__ import annotations

import polars as pl


class TestRunPipelineSmoke:
    def test_config_instantiation(self):
        from src.data_processing.components import Config

        cfg = Config()
        assert cfg.data_path is not None
        assert cfg.archive_name == "chembl_36_sqlite.tar.gz"

    def test_pipeline_components_integrate(self, mocker, tmp_path):
        fake_df = pl.DataFrame(
            {
                "canonical_smiles": ["CCO", "CC(=O)O"],
                "standard_value": [100.0, 50.0],
                "standard_units": ["nM", "nM"],
                "pchembl_value": [None, None],
                "standard_type": ["IC50", "IC50"],
                "target_chembl_id": ["CHEMBL220", "CHEMBL220"],
                "target_name": ["hERG", "hERG"],
            }
        )

        mocker.patch(
            "src.data_processing.components.pl.read_database_uri",
            return_value=fake_df,
        )
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("src.data_processing.components.Path.mkdir")

        from src.data_processing.components import (
            Config,
            DataDownloader,
            DataLoader,
            DataProcessor,
        )

        cfg = Config(data_path=tmp_path)
        downloader = DataDownloader(cfg)
        mocker.patch.object(downloader, "download_and_extract")

        loader = DataLoader(cfg)
        raw_df = loader.load_from_sqlite()
        assert raw_df.height == 2

        processor = DataProcessor(cfg)
        result = processor.process_data(raw_df)
        assert result.height > 0
        assert "pIC50" in result.columns

    def test_run_pipeline_module_imports(self):
        from src.data_processing import components

        assert hasattr(components, "Config")
        assert hasattr(components, "DataDownloader")
        assert hasattr(components, "DataLoader")
        assert hasattr(components, "DataProcessor")
