"""Clasificación de rachas de venta cero: normal, agotado, inactivo o pre-lanzamiento."""
import polars as pl


def clasificar_ceros(df: pl.DataFrame, ventana=56, p0_min=0.01, p_umbral=1e-4,
                     min_dias=3, p0_max=0.5, dias_inactivo=90) -> pl.DataFrame:
    """
    df: columnas id, date, sales (formato largo, días consecutivos por producto).
    Devuelve una fila por producto-día con columnas tipo y excluir.
    Criterio (igual que 02_baselines): probabilidad local de cero en los 56 días previos,
    agotado si la racha es improbable por azar en un producto que vende al menos la mitad
    de los días; inactivo si la racha dura 90 días o más.
    """
    base = (
        df.select(["id", "date", "sales"])
        .with_columns(pl.col("id").cast(pl.Utf8), pl.col("sales").cast(pl.Int64))
        .sort(["id", "date"])
        .with_columns(cero=pl.col("sales") == 0)
        .with_columns(
            p0_prev=pl.col("cero").cast(pl.Float64)
                      .rolling_mean(window_size=ventana, min_samples=28).shift(1).over("id"),
            acum=pl.col("sales").cum_sum().over("id"),
            grupo=(pl.col("cero") != pl.col("cero").shift(1).over("id"))
                  .fill_null(True).cast(pl.Int32).cum_sum().over("id"),
        )
    )
    rachas = (
        base.filter(pl.col("cero"))
        .group_by(["id", "grupo"])
        .agg(dias=pl.len(), inicio=pl.col("date").min(), fin=pl.col("date").max(),
             p0=pl.col("p0_prev").first(), pre_lanzamiento=(pl.col("acum").max() == 0))
        .with_columns(p_azar=pl.col("p0").clip(lower_bound=p0_min) ** pl.col("dias"))
        .with_columns(
            tipo=pl.when(pl.col("pre_lanzamiento")).then(pl.lit("pre_lanzamiento"))
            .when(pl.col("dias") >= dias_inactivo).then(pl.lit("inactivo"))
            .when((pl.col("dias") >= min_dias) & (pl.col("p0") <= p0_max) & (pl.col("p_azar") < p_umbral))
            .then(pl.lit("agotado"))
            .otherwise(pl.lit("normal"))
        )
    )
    marcados = (
        rachas.filter(pl.col("tipo") != "normal")
        .with_columns(date=pl.date_ranges("inicio", "fin"))
        .explode("date")
        .select(["id", "date", "tipo"])
    )
    return (
        base.select(["id", "date"])
        .join(marcados, on=["id", "date"], how="left")
        .with_columns(tipo=pl.col("tipo").fill_null("normal"))
        .with_columns(excluir=pl.col("tipo") != "normal")
    )