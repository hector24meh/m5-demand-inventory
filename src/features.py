"""Features para LightGBM. Todas usan información de 28 días atrás o más."""
import pandas as pd
import polars as pl

LAGS = [28, 35, 42, 49, 56]
CAT_FEATS = ["item_id", "dept_id", "event_type_1", "event_name_1"]
NUM_FEATS = ([f"lag_{l}" for l in LAGS] +
             ["rm_7", "rm_28", "rm_56", "rm_112", "rstd_28", "ceros_28", "excl_28",
              "dia_semana", "dia_mes", "mes", "semana_anio", "snap",
              "sell_price", "precio_rel", "precio_cambio", "edad"])
FEATS = CAT_FEATS + NUM_FEATS


def construir_features(dev: pl.DataFrame, mascara: pl.DataFrame) -> pd.DataFrame:
    """dev: ventas en formato largo (id como Utf8). mascara: columnas id, date, excluir."""
    feats = (
        dev.join(mascara.select(["id", "date", "excluir"]), on=["id", "date"], how="left")
        .with_columns(pl.col("excluir").fill_null(False))
        .sort(["id", "date"])
        .with_columns([pl.col("sales").shift(l).over("id").alias(f"lag_{l}") for l in LAGS])
        .with_columns(
            rm_7=pl.col("lag_28").rolling_mean(7).over("id"),
            rm_28=pl.col("lag_28").rolling_mean(28).over("id"),
            rm_56=pl.col("lag_28").rolling_mean(56).over("id"),
            rm_112=pl.col("lag_28").rolling_mean(112).over("id"),
            rstd_28=pl.col("lag_28").rolling_std(28).over("id"),
            ceros_28=(pl.col("lag_28") == 0).cast(pl.Float32).rolling_mean(28).over("id"),
            excl_28=pl.col("excluir").cast(pl.Float32).shift(28).rolling_mean(28).over("id"),
            dia_semana=pl.col("date").dt.weekday(),
            dia_mes=pl.col("date").dt.day(),
            mes=pl.col("date").dt.month(),
            semana_anio=pl.col("date").dt.week(),
            navidad=((pl.col("date").dt.month() == 12) & (pl.col("date").dt.day() == 25)).cast(pl.Int8),
            edad=(pl.col("date") - pl.col("date").min().over("id")).dt.total_days(),
            precio_rel=pl.col("sell_price")
                       / (pl.col("sell_price").cum_sum() / pl.col("sell_price").cum_count()).over("id"),
            precio_cambio=pl.col("sell_price") / pl.col("sell_price").shift(7).over("id") - 1,
        )
    )
    pdf = feats.select(["id", "date", "sales", "excluir", "navidad", *FEATS]).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"]).astype("datetime64[ns]")
    for c in CAT_FEATS:
        pdf[c] = pdf[c].astype(str).astype("category")
    pdf[NUM_FEATS] = pdf[NUM_FEATS].astype("float32")
    return pdf