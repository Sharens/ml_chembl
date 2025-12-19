import polars as pl

def compute_pIC50(df: pl.DataFrame) -> pl.DataFrame:
    """
    Oblicza pIC50 = -log10(M) dla wartości w nM.
    Dodaje kolumnę pIC50.
    """
    # Zamieniamy tylko rekordy, które mają jednostki "nM" i standard_value nie null
    df = df.with_columns(
        pl.when(
            (pl.col("standard_units") == "nM") & pl.col("standard_value").is_not_null()
        )
        .then(-(pl.col("standard_value") * 1e-9).log10())
        .otherwise(None)
        .alias("pIC50")
    )
    return df


def impute_units(df: pl.DataFrame, value_col="standard_value", units_col="standard_units") -> pl.DataFrame:
    """
    Imputuje brakujące standard_units na podstawie zakresu standard_value
    i dodaje kolumnę units_imputed.
    """
    df = df.with_columns([
        pl.lit(False).alias("units_imputed")
    ])

    # Maski: brak jednostki i wartość istnieje
    mask_missing = pl.col(units_col).is_null() & pl.col(value_col).is_not_null()
    mask_range = (pl.col(value_col) >= 0.01) & (pl.col(value_col) <= 1e6)

    df = df.with_columns([
        pl.when(mask_missing & mask_range)
        .then(pl.lit("nM"))
        .otherwise(pl.col(units_col))
        .alias(units_col),
        pl.when(mask_missing & mask_range)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("units_imputed")
    ])

    return df