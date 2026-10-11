from pathlib import Path
import polars as pl

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

STORE = "CA_1"
STATE = STORE.split("_")[0]
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAT_COLS = ID_COLS + ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

# 1) Ventas: filtrar tienda y pasar de ancho (d_1..d_1941) a largo
sales = (
    pl.scan_csv(RAW / "sales_train_evaluation.csv")
    .filter(pl.col("store_id") == STORE)
    .collect()
    .unpivot(index=ID_COLS, variable_name="d", value_name="sales")
    .with_columns(pl.col("sales").cast(pl.Int16))
)

# 2) Calendario: solo la columna SNAP del estado de la tienda
calendar = (
    pl.read_csv(RAW / "calendar.csv", try_parse_dates=True)
    .select(["d", "date", "wm_yr_wk", "wday", "month", "year",
             "event_name_1", "event_type_1", "event_name_2", "event_type_2",
             f"snap_{STATE}"])
    .rename({f"snap_{STATE}": "snap"})
    .with_columns(pl.col("snap").cast(pl.Int8), pl.col("wm_yr_wk").cast(pl.Int32))
)

# 3) Precios semanales de la tienda
prices = (
    pl.scan_csv(RAW / "sell_prices.csv")
    .filter(pl.col("store_id") == STORE)
    .collect()
    .with_columns(pl.col("wm_yr_wk").cast(pl.Int32), pl.col("sell_price").cast(pl.Float32))
)

df = (
    sales.join(calendar, on="d", how="left")
    .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
)

# Validación: filas sin precio = producto todavía no a la venta (no son ceros reales)
n_total = df.height
n_sin_precio = df.filter(pl.col("sell_price").is_null()).height
print(f"Filas totales: {n_total:,} | sin precio (se eliminan): {n_sin_precio:,} "
      f"({n_sin_precio / n_total:.1%})")

df = (
    df.filter(pl.col("sell_price").is_not_null())
    .with_columns([pl.col(c).fill_null("none") for c in CAT_COLS if c.startswith("event")])
    .with_columns([pl.col(c).cast(pl.Categorical) for c in CAT_COLS])
    .drop("d")
    .sort(["id", "date"])
)

df.write_parquet(OUT / f"m5_{STORE}.parquet")
print(df.shape, "|", df["date"].min(), "→", df["date"].max(), "|", df["id"].n_unique(), "series")


