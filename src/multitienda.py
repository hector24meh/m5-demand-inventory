"""Pipeline completo de una tienda: datos → detector de agotados → pronósticos → simulación (R, S).

Replica la lógica de prepare_data.py, 01_eda, 02_baselines y 06_ablacion_censura para poder
repetir la ablación (LightGBM, TSB con ventas, TSB con demanda corregida, seasonal naive) en
cualquier tienda de M5. AutoETS se omite: tarda ~20 min por tienda.
"""
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import TSB, SeasonalNaive

from censura import clasificar_ceros
from features import CAT_FEATS, FEATS, construir_features
from simulate import simular_rs

R, L = 7, 7
PROT = R + L - 1
N_CALIB, N_SIM = 13, 26
H_ANUAL, MARGEN = 0.25, 0.30
WARMUP = 14
ALPHAS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
N_BINS = 5
H_PRUEBA, VENTANAS_BT, MIN_HIST = 28, 3, 180
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1,
              n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              n_jobs=-1, verbose=-1)
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAT_COLS = ID_COLS + ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


# ---------- Datos (prepare_data.py) ----------
def preparar(raw: Path, tienda: str) -> pl.DataFrame:
    estado = tienda.split("_")[0]
    ventas = (pl.scan_csv(raw / "sales_train_evaluation.csv").filter(pl.col("store_id") == tienda).collect()
              .unpivot(index=ID_COLS, variable_name="d", value_name="sales")
              .with_columns(pl.col("sales").cast(pl.Int16)))
    calendario = (pl.read_csv(raw / "calendar.csv", try_parse_dates=True)
                  .select(["d", "date", "wm_yr_wk", "wday", "month", "year",
                           "event_name_1", "event_type_1", "event_name_2", "event_type_2", f"snap_{estado}"])
                  .rename({f"snap_{estado}": "snap"})
                  .with_columns(pl.col("snap").cast(pl.Int8), pl.col("wm_yr_wk").cast(pl.Int32)))
    precios = (pl.scan_csv(raw / "sell_prices.csv").filter(pl.col("store_id") == tienda).collect()
               .with_columns(pl.col("wm_yr_wk").cast(pl.Int32), pl.col("sell_price").cast(pl.Float32)))
    return (ventas.join(calendario, on="d", how="left")
            .join(precios, on=["store_id", "item_id", "wm_yr_wk"], how="left")
            .filter(pl.col("sell_price").is_not_null())
            .with_columns([pl.col(c).fill_null("none") for c in CAT_COLS if c.startswith("event")])
            .with_columns([pl.col(c).cast(pl.Categorical) for c in CAT_COLS])
            .drop("d").sort(["id", "date"]))


# ---------- Segmentación (01_eda) y universo (02_baselines) ----------
def clase_demanda(df: pl.DataFrame) -> pl.DataFrame:
    return (df.group_by("id")
            .agg(n=pl.len(), nz=(pl.col("sales") > 0).sum(),
                 mean_nz=pl.col("sales").filter(pl.col("sales") > 0).mean(),
                 std_nz=pl.col("sales").filter(pl.col("sales") > 0).std())
            .filter(pl.col("nz") > 1)
            .with_columns(adi=pl.col("n") / pl.col("nz"), cv2=(pl.col("std_nz") / pl.col("mean_nz")) ** 2)
            .with_columns(sb_class=pl.when((pl.col("adi") < 1.32) & (pl.col("cv2") < 0.49)).then(pl.lit("smooth"))
                          .when(pl.col("adi") < 1.32).then(pl.lit("erratic"))
                          .when(pl.col("cv2") < 0.49).then(pl.lit("intermittent"))
                          .otherwise(pl.lit("lumpy")))
            .select(id=pl.col("id").cast(pl.Utf8), sb_class="sb_class"))


def series_validas(dev: pl.DataFrame, test_start: pd.Timestamp) -> list[str]:
    first_cutoff = test_start - timedelta(days=H_PRUEBA * VENTANAS_BT + 1)
    chk = (dev.group_by("id").agg(inicio=pl.col("date").min(), fin=pl.col("date").max(), n=pl.len())
           .with_columns(huecos=(pl.col("fin") - pl.col("inicio")).dt.total_days() + 1 - pl.col("n")))
    limite = (first_cutoff - timedelta(days=MIN_HIST)).date()
    return chk.filter((pl.col("inicio") <= limite) & (pl.col("huecos") == 0))["id"].cast(pl.Utf8).to_list()


# ---------- Simulación (04 / 06) ----------
def suma_por_origen(M: pd.DataFrame, origenes) -> pd.DataFrame:
    return pd.DataFrame({o: M.loc[o + timedelta(days=1): o + timedelta(days=PROT)].sum() for o in origenes}).T


def correr_tienda(df: pl.DataFrame, log=print) -> dict:
    df = df.with_columns(pl.col("id").cast(pl.Utf8))
    test_start = pd.Timestamp(df["date"].max()) - timedelta(days=H_PRUEBA - 1)
    dev_end = test_start - timedelta(days=1)
    last_review = dev_end - timedelta(days=PROT)
    reviews_sim = [last_review - timedelta(days=7 * k) for k in range(N_SIM)][::-1]
    sim_start = reviews_sim[0]
    last_calib = sim_start - timedelta(days=PROT + 1)
    origins_calib = [last_calib - timedelta(days=7 * k) for k in range(N_CALIB)][::-1]
    train_end = origins_calib[0]
    origenes = pd.date_range(origins_calib[0], last_review, freq="7D")

    dev_todo = df.filter(pl.col("date") <= dev_end.date())
    mascara = clasificar_ceros(dev_todo.select(["id", "date", "sales"])).select(["id", "date", "tipo", "excluir"])
    ids_validos = series_validas(dev_todo, test_start)
    clase = clase_demanda(df)

    dev = dev_todo.filter(pl.col("id").is_in(ids_validos))
    inicio = dev.group_by("id").agg(inicio=pl.col("date").min())
    fuera = (mascara.filter((pl.col("date") > train_end.date()) & pl.col("tipo").is_in(["inactivo", "pre_lanzamiento"]))
             ["id"].unique().to_list())
    universo = sorted(inicio.filter(pl.col("inicio") <= (train_end - timedelta(days=112)).date())
                      .filter(~pl.col("id").is_in(fuera))["id"].to_list())

    d = (dev.filter(pl.col("id").is_in(universo))
         .join(mascara, on=["id", "date"], how="left")
         .with_columns(pl.col("excluir").fill_null(False), pl.col("tipo").fill_null("normal"),
                       ventas=pl.col("sales").cast(pl.Float64))
         .sort(["id", "date"])
         .with_columns(limpia=pl.when(pl.col("excluir")).then(pl.lit(None)).otherwise(pl.col("ventas")))
         .with_columns(num=pl.col("limpia").fill_null(0).rolling_sum(28).shift(1).over("id"),
                       den=(~pl.col("excluir")).cast(pl.Float64).rolling_sum(28).shift(1).over("id"),
                       media_prod=pl.col("limpia").mean().over("id"))
         .with_columns(imp=pl.when(pl.col("den") > 0).then(pl.col("num") / pl.col("den")).otherwise(pl.col("media_prod")))
         .with_columns(demanda=pl.when(pl.col("tipo") == "agotado").then(pl.col("imp")).otherwise(pl.col("ventas")))
         .with_columns(demanda=pl.when((pl.col("date").dt.month() == 12) & (pl.col("date").dt.day() == 25))
                       .then(0.0).otherwise(pl.col("demanda"))))

    # LightGBM entrenado una vez, sin días censurados
    pdf = construir_features(dev, mascara)
    tr = pdf[(pdf["date"] <= train_end) & (~pdf["excluir"]) & (pdf["navidad"] == 0)]
    modelo = lgb.LGBMRegressor(**PARAMS).fit(tr[FEATS], tr["sales"], categorical_feature=CAT_FEATS)
    te = pdf[(pdf["date"] > train_end) & (pdf["id"].isin(universo))].copy()
    te["LGB"] = np.clip(modelo.predict(te[FEATS]), 0, None)
    fechas = pd.date_range(origenes[0] + timedelta(days=1), dev_end)
    P_lgb = te.pivot(index="date", columns="id", values="LGB").reindex(index=fechas, columns=universo)

    # Clásicos con ventas y con demanda corregida (n_jobs=1: evita bloqueos con Polars)
    def cv(columna, modelos):
        Y = d.select(unique_id="id", ds="date", y=pl.col(columna).cast(pl.Float32)).to_pandas()
        Y["ds"] = pd.to_datetime(Y["ds"]).astype("datetime64[ns]")
        out = StatsForecast(models=modelos, freq="D", n_jobs=1).cross_validation(
            df=Y, h=PROT, step_size=7, n_windows=len(origenes))
        if "unique_id" not in out.columns:
            out = out.reset_index()
        out["cutoff"] = pd.to_datetime(out["cutoff"]).astype("datetime64[ns]")
        nombres = [c for c in out.columns if c not in ("unique_id", "ds", "cutoff", "y")]
        out[nombres] = out[nombres].clip(lower=0)
        return {m: out.groupby(["cutoff", "unique_id"])[m].sum().unstack().reindex(index=origenes, columns=universo)
                for m in nombres}

    F = {"LGB": suma_por_origen(P_lgb, origenes)}
    F.update(cv("ventas", [SeasonalNaive(season_length=7), TSB(alpha_d=0.1, alpha_p=0.1)]))
    F["TSB_imp"] = cv("demanda", [TSB(alpha_d=0.1, alpha_p=0.1)])["TSB"]

    pdd = d.select(["id", "date", "demanda", "sell_price"]).to_pandas()
    pdd["date"] = pd.to_datetime(pdd["date"]).astype("datetime64[ns]")
    D_full = pdd.pivot(index="date", columns="id", values="demanda").reindex(index=fechas, columns=universo)
    precio_full = pdd.pivot(index="date", columns="id", values="sell_price").reindex(index=fechas, columns=universo)
    A = suma_por_origen(D_full, origenes)
    calib = [o for o in origenes if o <= last_calib]

    clase_s = clase.to_pandas().set_index("id")["sb_class"].reindex(universo)
    sim_fechas = pd.date_range(sim_start, dev_end)
    D_sim, precio = D_full.loc[sim_fechas].to_numpy(), precio_full.loc[sim_fechas].to_numpy()
    revision = sim_fechas.isin(reviews_sim)
    idx_rev = np.where(revision)[0]

    def metricas(res):
        s = slice(WARMUP, None)
        dem, vend, perd, inv, p = (x[s] for x in (D_sim, res["vendido"], res["perdido"], res["inventario"], precio))
        c_mant, c_falt = (inv * p * H_ANUAL / 365).sum(), (perd * p * MARGEN).sum()
        return {"fill_rate": vend.sum() / dem.sum(), "inv_valor_prom": (inv * p).sum(axis=1).mean(),
                "costo_mantener": c_mant, "costo_faltante": c_falt, "costo_total": c_mant + c_falt}

    filas = []
    for m, Fm_all in F.items():
        Fc = Fm_all.loc[calib]
        z = (A.loc[calib] - Fc) / np.sqrt(Fc + 1)
        zl = pd.DataFrame({"id": np.tile(universo, len(calib)), "f": Fc.to_numpy().ravel(), "z": z.to_numpy().ravel()})
        zl["clase"] = zl["id"].map(clase_s)
        q_clase = zl.groupby("clase")["z"].quantile(ALPHAS).unstack()
        edges = np.unique(np.quantile(zl["f"], np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        zl["bin"] = pd.cut(zl["f"], edges, labels=False).astype(int)
        tabla_q = zl.groupby(["clase", "bin"])["z"].quantile(ALPHAS).unstack()

        Fm = Fm_all.loc[reviews_sim].to_numpy()
        bins = pd.cut(Fm.ravel(), edges, labels=False).astype(int)
        claves = pd.MultiIndex.from_arrays([np.tile(clase_s.to_numpy(), len(reviews_sim)), bins])
        for a in ALPHAS:
            qa = tabla_q[a].reindex(claves).to_numpy().reshape(Fm.shape)
            qa = np.where(np.isnan(qa), np.tile(clase_s.map(q_clase[a]).to_numpy(), (len(reviews_sim), 1)), qa)
            S = np.zeros_like(D_sim)
            S[idx_rev] = np.maximum(0.0, Fm + qa * np.sqrt(Fm + 1))
            res = simular_rs(D_sim, S, revision, L, inv_inicial=np.ceil(S[0]))
            filas.append({"modelo": m, "alpha": a, **metricas(res)})

    simo = [o for o in origenes if o >= sim_start]
    return {
        "tabla": pd.DataFrame(filas),
        "productos": len(universo),
        "agotado_pct": float((mascara["tipo"] == "agotado").mean() * 100),
        "imputado_pct": float((d.filter(pl.col("date") > train_end.date())["tipo"] == "agotado").mean() * 100),
        "sesgo_sim_pct": {m: float((F[m].loc[simo] - A.loc[simo]).sum().sum() / A.loc[simo].sum().sum() * 100) for m in F},
    }
