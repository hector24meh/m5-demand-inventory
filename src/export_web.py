"""Exporta el resumen del proyecto M5 a JSON para la página del portafolio.
Uso: uv run python src/export_web.py   (después de la Celda 11 de 04_inventario.ipynb)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
PROC = ROOT / "data" / "processed"
OUT = Path.home() / "Documents" / "portfolio" / "public" / "data" / "m5"
OUT.mkdir(parents=True, exist_ok=True)


def limpiar_columnas(df):
    return df.rename(columns=lambda c: c.replace("%", "pct").lower())


def registros(df, dec=4):
    return json.loads(limpiar_columnas(df).round(dec).to_json(orient="records"))


# --- Inventario ---
inv = pd.read_csv(REP / "simulacion_inventario_volumen_CA_1.csv")


def inventario_en(modelo, fr):
    c = inv[inv["modelo"] == modelo].sort_values("fill_rate")
    if not (c["fill_rate"].min() <= fr <= c["fill_rate"].max()):
        return None
    return float(np.interp(fr, c["fill_rate"], c["inv_valor_prom"]))


ahorro_por_fr = []
for fr in [0.92, 0.95, 0.97, 0.98]:
    i_lgb, i_tsb = inventario_en("LGB", fr), inventario_en("TSB", fr)
    ahorro_por_fr.append({"fill_rate": fr, "inv_lgb": i_lgb, "inv_tsb": i_tsb,
                          "ahorro_pct": round((1 - i_lgb / i_tsb) * 100, 1) if i_lgb and i_tsb else None})

opt = inv.loc[inv.groupby("modelo")["costo_total"].idxmin()].set_index("modelo")
pf = pd.read_csv(REP / "metricas_prueba_final_CA_1.csv").set_index("modelo")
bt = pd.read_csv(REP / "metricas_lgb_vs_baselines_CA_1.csv")
meta = json.loads((OUT / "meta_simulacion.json").read_text())

kpis = {
    "ahorro_inventario_fr95_pct": next(a["ahorro_pct"] for a in ahorro_por_fr if a["fill_rate"] == 0.95),
    "ahorro_costo_vs_tsb_pct": round((1 - opt.loc["LGB", "costo_total"] / opt.loc["TSB", "costo_total"]) * 100, 1),
    "ahorro_costo_vs_sn_pct": round((1 - opt.loc["LGB", "costo_total"] / opt.loc["SeasonalNaive", "costo_total"]) * 100, 1),
    "fill_rate_optimo_lgb": round(float(opt.loc["LGB", "fill_rate"]), 4),
    "fill_rate_optimo_tsb": round(float(opt.loc["TSB", "fill_rate"]), 4),
    "rmsse_prueba_lgb": round(float(pf.loc["LGB", "RMSSE_limpio"]), 3),
    "rmsse_prueba_autoets": round(float(pf.loc["AutoETS", "RMSSE_limpio"]), 3),
    "sesgo_prueba_lgb_pct": round(float(pf.loc["LGB", "Sesgo_%_limpio"]), 1),
    "sesgo_prueba_autoets_pct": round(float(pf.loc["AutoETS", "Sesgo_%_limpio"]), 1),
    "sesgo_prueba_tsb_pct": round(float(pf.loc["TSB", "Sesgo_%_limpio"]), 1),
    **meta,
}

# --- EDA y censura ---
seg = pl.read_parquet(PROC / "segmentacion_CA_1.parquet")
mascara = pl.read_parquet(PROC / "mascara_CA_1.parquet")
eda = {
    "clases_demanda": dict(seg.group_by("sb_class").len().iter_rows()),
    "matriz_abc_xyz": [{"abc": a, "xyz": x, "productos": n}
                       for a, x, n in seg.group_by(["abc", "xyz"]).len().sort(["abc", "xyz"]).iter_rows()],
    "censura_pct": {t: round(n / mascara.height * 100, 2)
                    for t, n in mascara.group_by("tipo").len().iter_rows()},
}

resumen = {
    "kpis": kpis,
    "metricas_backtest": registros(bt),
    "prueba_final": registros(pf.reset_index()),
    "prueba_final_abc": registros(pd.read_csv(REP / "prueba_final_por_abc_CA_1.csv")
                                  .rename(columns={"Unnamed: 0": "abc"})),
    "tradeoff": registros(inv[["modelo", "alpha", "fill_rate", "inv_valor_prom",
                               "dias_cobertura", "costo_total"]]),
    "ahorro_por_fill_rate": ahorro_por_fr,
    "sensibilidad": registros(pd.read_csv(REP / "sensibilidad_costos_CA_1.csv")),
    "eda": eda,
}

(OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1))
print(f"resumen.json → {OUT} ({(OUT / 'resumen.json').stat().st_size / 1024:.0f} KB)")
print(json.dumps(kpis, indent=2, ensure_ascii=False))